
# Implementing the Fourier-domain PTA likelihood of (S. Valtolina, R. van Haasteren, 2025)
# Source: https://journals.aps.org/prd/abstract/10.1103/s3gy-km61
# In what follows, we refer to the paper as 'vvh25'

import dataclasses
from typing import Optional
import functools
import re
from tqdm import tqdm

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp

import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer

from .. import (likelihood, prior, signals, matrix)
from .. import partial

from .nanograv_single_pulsar_outlier import _lookup_prior
from .spectral_covs import *

from .. import priordict_standard
newdict = {'(.*_)?red_noise_coefficients\\(([0-9]*)\\)': [-100, 100],
           '(.*_)?log10k': [-9.0, -4.0]}

priordict_standard.update({
    '(.*_)?red_noise_coefficients\\(([0-9]*)\\)': [-100, 100], 
    '(.*_)?log10k': [-9.0, -4.0]}) # for flat-tail powerlaw spectrum


@dataclasses.dataclass
class PulsarFourierSummary:
    """
    Per-pulsar containers for the Fourer-domain based PTA analysis,
    with each pulsar object represented as an instance of the class.
    
    Step 1 involves the per-pulsar analysis, where regularized quantities
    are stored evaluated at a reference hyperpameter eta0. All regularizing
    quantities are then stored. 
    
    Whenever all necesary fields have been populated, the per-pulsar summary objects
    are ready to enter the joint analysis.
    """
    
    name: str
    psl: object # discovery PulsarLikelihood object
    eta0: dict  # hyperparameter regularizer
    
    phi0_inv: jnp.ndarray # spectral prior cov. matrix evaluated at eta0
    logdet_phi0: float  # log determinant of phi_0

    # PTA conditional quantiites evalauted with regularizer (r)
    ahat0r: jnp.ndarray # conditional mean of Fourier coeff. at eta0
    L0r: jnp.ndarray # cholesky of conditional cov. of Fourier coeff. at eta0

    # regularizing quantities needed for step 2
    # by default set to None and populated
    b0: jnp.ndarray = None  # = Sigma0^{-1} \ahat_0 
    TtNT: jnp.ndarray = None # = Sigma0^{-1} - Ph0^{-1}
    log_const0: Optional[float] = None # any constant offsets (will not affect sampler hence optional)
    
    # Gaussian approximation quantities
    # after marginalizing out quantities not represented in the Fourier domain
    ahat0: jnp.ndarray = None  # mean estimate to marginalized Fourier coefficients from step 1
    Sigma0: jnp.ndarray = None  # cov. estimate to marignalized Fourier coefficients from step 1
    L0: jnp.ndarray = None  # cholesky of Sigma0

    # container for numpyro samples as outputed of sampler in step 1
    samples: dict = None  # can also be set manually which updates theta_samples (see below)

    @property
    def n_coeff(self):
        """Number of Fourier coefficients per pulsar"""
        return self.ahat0r.shape[0]

    @property
    def rn_slice(self):
        """Picks out Fourier coefficients from full coefficient vector"""
        rn_key = [k for k in self.psl.N.index.keys() if 'red_noise' in k][0]
        return self.psl.N.index[rn_key]

    @property
    def is_ready_for_step2(self):
        """Checks if per-pulsar summaries are ready to enter the second step
        NOTE: log_const0 is not a required quantity since it does not affect final result 
        """
        required = {'ahat0': self.ahat0, 'L0': self.L0, 'b0': self.b0, 'TtNT': self.TtNT}
        missing = [k for k, v in required.items() if v is None]
        if missing:
            print(f"Pulsar {self.name} is missing the following quantities: {missing}")
            return False
        return True

    @property
    def theta_samples(self):
        """
        Populates the step-1 samples from sampler not incl. the Fourier coefficients
        This is later used in compute_zero_quantities to get a Gaussian approximation.
        """
        if self.samples is None:
            return None
        return {p: self.samples[p] for p in self.samples.keys() if p not in ('y', 'a')}


def latent_space_transformation(func,priordict=newdict):
    """
    Latent space transformation used for both learning a normalizing flow
    on an unconstrained latent space and for HMC sampler.
    
    Hyperparameters are mapped from real line to prior bounds using tanh-transformation.
    NOTE: only works if prior is UNIFORM.
    
    Fourier coefficients are expressed in non-centered parametrization
    using a = ahat + L @ y
    
    """
    priordict = {**priordict_standard, **priordict}

    slices, offset = [], 0

    for par in func.params:
        if '(' in par:
            l = int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1
            slices.append(slice(offset, offset+l))
            offset = offset + l
        else:
            slices.append(offset)
            offset = offset + 1

    a, b = [], []
    columns = []
    for par, slice_ in zip(func.params, slices):
        for pname, prange in priordict.items():
            if re.match(pname, par):
                therange = prange
                break
        else:
            raise KeyError(f"No known prior for {par}.")

        if '(' in par:
            root = par[:par.index('(')]
            l = int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1

            for i in range(l):
                columns.append(f'{root}[{i}]')
                a.append(therange[0])
                b.append(therange[1])
        else:
            columns.append(par)
            a.append(therange[0])
            b.append(therange[1])
    a, b = matrix.jnparray(a), matrix.jnparray(b)

    def to_dict_and_jacobian(ys, ahat, L):
        """Maps variables from latent space to real space while accounting
        for Jacobian of the transformation."""
        
        hyper_pars = [p for p in func.params if "coefficients" not in p]
        coeff_pars = [p for p in func.params if "coefficients" in p]
        len_hyper = len(hyper_pars)

        if len_hyper > 0:
            # tanh transforamtion (NOTE! Assumes prior on hyperpars is uniform)
            ys_hyper = ys[:len_hyper]
            xs_hyper = 0.5 * (b[:len_hyper] + a[:len_hyper] + 
                            (b[:len_hyper] - a[:len_hyper]) * jnp.tanh(ys_hyper))
            hyper_jacobian = jnp.sum(jnp.log(2.0) - 2.0 * jnp.logaddexp(ys_hyper, -ys_hyper))
            hyper_dict = dict(zip(hyper_pars, jnp.array(xs_hyper).T))
        else:
            # in case no hyperparameters are involved
            hyper_jacobian = 0.0
            hyper_dict = {}

        # decentering for Fourier coefficients
        jac = jnp.sum(jnp.log(jnp.diag(L)))
        coeff_idx = func.params.index(coeff_pars[0])  # NOTE! coeff_pars assumes 0th index RN coeffs.
        xs_coeff = L @ ys[slices[coeff_idx]] + ahat
        hyper_dict.update({coeff_pars[0]: xs_coeff})

        return hyper_dict, jac + hyper_jacobian

        
    def transformed(ys, ahat, L):
        """Loglikelihood evalauted in latent space incl. Jacobian"""
        mydict, jac = to_dict_and_jacobian(ys, ahat, L)
        return func(mydict) + jac
    
    transformed.params = func.params
    transformed.to_dict_and_jacobian = to_dict_and_jacobian
    transformed.a_bounds = a
    transformed.b_bounds = b
    transformed.columns = columns
    
    return transformed

#### Marginalized Fourier likelihoods ####

def make_fourier_model(psrs,
                       Tspan=None,
                       psd=None,
                       components: int = 30,
                       rn_name: str = "red_noise",
                       noisedict=None,
                       ecorr: bool = True):

    """
    Builds discovery PulsarLikelihood per pulsar for Fourier model.
    
    If psd and noisedict are single objects, then they are shared across all pulsars
    included in the array. noisedict = None leaves the white noise free.
    """
    
    if Tspan is None:
        Tspan = signals.getspan(psrs)
    if psd is None:
        psd = signals.powerlaw

    n = len(psrs)
    # either single noisedict applied to all pulsar, or pulsar-specific noisedicts
    if noisedict is None:
        noisedict_list = [{}] * n  # free WN parameters
    elif isinstance(noisedict, dict):
        noisedict_list = [noisedict] * n  
    elif isinstance(noisedict, list):
        if len(noisedict) != n:
            raise ValueError(
                f"noisedict list length  with {len(noisedict)} noisedicts does not match "
                f"{n} number of pulsars.")
        noisedict_list = noisedict  
    else:
        raise ValueError("noisedict must be a dict, list of dicts, or None.")

  
    # either single PSD applied to all pulsar, or pulsar-specific psds
    psd_list = psd if isinstance(psd, list) else [psd] * n

    if len(psd_list) != n:
        raise ValueError(
            f"psd list length {len(psd_list)} mismatch with number of pulsars {n}.")

    return [likelihood.PulsarLikelihood([psr.residuals, 
                                         signals.makegp_timing(psr, svd=True),
                                         signals.makenoise_measurement(psr, noisedict=nd, ecorr=ecorr),
                                         signals.makegp_fourier(psr, p, components, T=Tspan, name=rn_name)])
                                for psr, p, nd in zip(psrs, psd_list, noisedict_list)]
    

def log_fourier_likelihood(rho, b, phi_func, TtNT, log_const0):
    """
    PTA Fourier-domain log-likelihood marginalized over Fourier coefficients
    in accordance with equation (22) of vvh25
    
    Used for SPNA and HD. 
    
    inputs:
    rho: dictionary of spectral hyperparameters (e.g. log10_A, gamma)
    phi_func: a function that maps rho to phi^{-1}, log | det phi |.   
    TtNT:  Sigma0^{-1}-phi0^{-1}
    b: Sigma_0^{-1} ahat_0, can be precomputed in step 1
    log_const0: constant offset which does not affect end result
    
    computes: Sigma^{-1}: Sigma0^{-1}-phi0^{-1}.
    
    returns log_const0 + 0.5*b^T Sigma b + 0.5 log | det Sigma| - 0.5 * log | det phi| 
    """
    
    phi_inv, logdet_phi = phi_func(rho)

    Sigma_inv = TtNT + phi_inv # Sigma_inv from (23) of vvh25
    L_inv = jnp.linalg.cholesky(Sigma_inv)
    
    log_det_Sigma = -2.0 * jnp.sum(jnp.log(jnp.diag(L_inv)))
    mu = jsp.linalg.cho_solve((L_inv, True), b)
    quad_b = b @ mu

    return log_const0 + 0.5 * quad_b + 0.5 * log_det_Sigma - 0.5 * logdet_phi


def log_fourier_likelihood_batched(rho, b, phi_func, TtNT, log_const0):
    
    """
    Same log-likelihood implementation as 'log_fourier_likelihood' except that it is used for CURN.
    Allows for batched Choleskys over pulsars. As long as common process
    does not introduce pulsar correlations, this batched model can be used.
    
    """
    phi_inv_diags, logdet_phi = phi_func(rho)

    sigma_inv = TtNT + jax.vmap(jnp.diag)(phi_inv_diags)
    L_inv = jax.vmap(jnp.linalg.cholesky)(sigma_inv)

    log_det_sigma = -2.0 * jnp.sum(jax.vmap(lambda L: jnp.sum(jnp.log(jnp.diag(L))))(L_inv))

    mu = jax.vmap(lambda L_p, b_p: jsp.linalg.cho_solve((L_p, True), b_p))(L_inv, b)
    quad_b = jnp.sum(jax.vmap(jnp.dot)(b, mu))

    return log_const0 + 0.5 * quad_b + 0.5 * log_det_sigma - 0.5 * logdet_phi

#### Joint Fourier likelihoods ####


def log_fourier_joint_batched(rho, xi, b, phi_func, TtNT):
      
    """
    Joint (xi, eta) sampler corresponding to equation (22) in vvh25. Allows for batching over pulsars.
    Used for CURN.
    """
    
    phi_inv_diags, logdet_phi = phi_func(rho)
    sigma_inv = TtNT + jax.vmap(jnp.diag)(phi_inv_diags)
    L_sinv = jax.vmap(jnp.linalg.cholesky)(sigma_inv)

    ahat = jax.vmap(lambda L_p, b_p: jsp.linalg.cho_solve((L_p, True), b_p))(L_sinv, b)
    a = ahat + jax.vmap(lambda L, xis: jsp.linalg.solve_triangular(L.T, xis, lower=False))(L_sinv, xi)

    quad_a = jnp.sum(jax.vmap(lambda sinv_p, a_p: a_p @ sinv_p @ a_p)(sigma_inv, a))
    linear_a = jnp.sum(jax.vmap(jnp.dot)(b, a))
    log_det_L = -jnp.sum(jax.vmap(lambda L: jnp.sum(jnp.log(jnp.diag(L))))(L_sinv))

    logL = -0.5 * quad_a + linear_a + log_det_L - 0.5 * logdet_phi 
    return logL, a


def log_jointFourierHD_dCURN(rho, xi, b, b_p, phi_hd_func, TtNT, n2_block, npsr):
    
    '''
    Joint (xi, eta) sampler corresponding to equation (22) in vvh25 for HD.
     
    Returns likelihood evaluation and Fourier coefficients using CURN decentering
    which allows for batching across pulsars. 
    
    '''
    
    phi_inv_hd, logdet_phi = phi_hd_func(rho)
    Sigma_inv = TtNT + phi_inv_hd

    # CURN decentering
    Sigma_inv_CURN = jnp.diagonal(Sigma_inv.reshape(npsr, n2_block, npsr, n2_block), axis1=0, axis2=2).transpose(2, 0, 1)
    L_sinv_CURN = jax.vmap(jnp.linalg.cholesky)(Sigma_inv_CURN)
    xi_p = xi.reshape(npsr, n2_block)
    
    ahat_p = jax.vmap(lambda L, bs: jsp.linalg.cho_solve((L, True), bs))(L_sinv_CURN, b_p)
    a = (ahat_p + jax.vmap(lambda L_p, xis: jsp.linalg.solve_triangular(L_p.T, xis, lower=False))(L_sinv_CURN, xi_p)).reshape(-1)
        
    # Jacobian for decentering transf.
    logdet_L_CURN = -jnp.sum(jnp.log(jax.vmap(jnp.diag)(L_sinv_CURN)))
    
    quad_a = a @ Sigma_inv @ a
    linear_a = a @ b # = a^T Sigma^{-1} ahat = a^T \Sigma^{-1} \Sigma Sigma_0^{-1} ahat0 = a^T b

    logL = -0.5 * quad_a + linear_a + logdet_L_CURN - 0.5 * logdet_phi 
    return logL, a



#### Step 1 functions ####

def compute_zero_quantities(summaries,
                            noisedict_list=None,
                            theta_samples_list=None):
    """
    Computes the regularizing (subscript-zero) quantities of vvh25 at eta0 and
    fully prepares each per-pulsar summary for the joint analysis in the second step of vvh25.


    Populates each per-pulsar summary with:

    1. ahat0, Sigma0 (the regularizing conditional mean and covariance)
        1.1 Fixed WN: ahat0, Sigma0 are taken directly from the conditional
            evaluated at the pulsar's noisedict.
        1.2 Free WN: the conditional is evaluated at each theta sample, and
            ahat0, Sigma0 follow from the law of total expectation / covariance,
            eq. (19) of vvh25. 
            It is up to the user to supply the summaries with the theta_samples
            which are marginalized over in equation (18) of vvh25
            
        NOTE: fixed theta vs. marginalized theta can be set on a pulsar-by-pulsar basis.

    2. b0 = Sigma0_inv @ ahat0
    3. L0 = chol(Sigma0)
    4. TtNT = Sigma0_inv - phi0_inv
    5. log_const0  (constant offset; does not affect the sampler)
    
    """

    n = len(summaries)

    if noisedict_list is None and theta_samples_list is None:
        raise ValueError("At least one of noisedict_list or theta_samples_list must be provided.")

    noisedict_list = noisedict_list if noisedict_list is not None else [None] * n
    theta_samples_list = theta_samples_list if theta_samples_list is not None else [None] * n

    for summary, nd, theta_samples in zip(summaries, noisedict_list, theta_samples_list):

        rn_sl = summary.rn_slice # red-noise block slice. NOTE: the rn_slice is needed 
                                # if e.g. ECORR is treated as gp process
                                # to extract the RN coefficients only
                                
        fixed_wn = nd is not None and len(nd) > 0 # per-pulsar decision
        if fixed_wn:
            
            # ahat0, cf_inv is determined deterministically through the conditional
            ahat0_full, cf_inv = summary.psl.conditional(nd)
            # NOTE: cf_inv returns the cholesky of Sigma0_inv
            # Sigma0 = (cf_inv cf_inv^T)^{-1}
            sigma0_full = jsp.linalg.cho_solve((cf_inv[0], True), jnp.eye(cf_inv[0].shape[0]))
            L0_full = jsp.linalg.cholesky(sigma0_full, lower=True)
            
            # rebuild sigma0_ivn for TtNT later
            sigma0_inv_full = cf_inv[0] @ cf_inv[0].T
            logdet_sigma0_inv = 2.0 * jnp.sum(jnp.log(jnp.diag(cf_inv[0]))) # cf_inv lower triangular

            # only pick out the Fourier coefficient blocks
            ahat0 = ahat0_full[rn_sl]
            sigma0 = sigma0_full[rn_sl, :][:, rn_sl]
            L0 = L0_full[rn_sl, :][:, rn_sl]
            sigma0_inv = sigma0_inv_full[rn_sl, :][:, rn_sl]

        else:
            
            # read in theta samples from current pulsar
            theta_param_names = [p for p in theta_samples.keys() if 'red_noise' not in p and 'coefficients' not in p]
            N_s = len(next(iter(theta_samples.values())))
            
            ahat_tilde_list  = []
            Sigma_tilde_list = []

            # computes the per-theta sample conditional mean ahat_tilde (theta) and covariance Sigma_tilde (theta)
            for j in tqdm(range(N_s), desc = f"{summary.name}"):
                nd_j = {p: float(theta_samples[p][j]) for p in theta_param_names} # read in current noisedict from theta_samples
                ahat_tilde, cf_inv_tilde = summary.psl.conditional(nd_j)    # evalaute conditional at given theta
                
                sigma_tilde = jsp.linalg.cho_solve((cf_inv_tilde[0], True),
                                  jnp.eye(cf_inv_tilde[0].shape[0]))
                ahat_tilde_list.append(np.array(ahat_tilde[rn_sl]))
                Sigma_tilde_list.append(np.array(sigma_tilde[rn_sl, :][:, rn_sl]))

            ahat_tilde_array  = jnp.array(ahat_tilde_list)
            Sigma_tilde_array = jnp.array(Sigma_tilde_list)

            # equation (4.14) of thesis
            ahat0 = jnp.mean(ahat_tilde_array, axis=0) # law of total expectation
            sigma0 = (jnp.mean(Sigma_tilde_array, axis=0) + jnp.cov(ahat_tilde_array.T, bias=False))    # law of total covariance
            
            L0 = jnp.linalg.cholesky(sigma0)
            sigma0_inv = jsp.linalg.cho_solve((L0, True), jnp.eye(sigma0.shape[0]))
            logdet_sigma0_inv = -2.0 * jnp.sum(jnp.log(jnp.diag(L0)))

        # additional zero quantities that can be precomputed of step 1
        b0 = sigma0_inv @ ahat0 
        TtNT = sigma0_inv - summary.phi0_inv 

        # the log-constant computed contains all constant factors of equation (22) in vvh25
        quad0 = float(ahat0 @ b0) 
        log_const0 = 0.5 * (float(logdet_sigma0_inv) - quad0 + summary.logdet_phi0)

        # writing to per-pulsar summaries
        summary.ahat0 = ahat0
        summary.Sigma0 = sigma0
        summary.L0 = L0
        summary.b0 = b0
        summary.TtNT = TtNT
        summary.log_const0 = log_const0
        
def make_step1_model(summary, priordict, ahat_l = None, L_l = None):
    
       
    """ 
    Construct a numpyro model that can be used to generate samples (theta, a) from
    
                    p(delta t \mid a, theta) p(a \mid \eta_0) p(theta)
    
    see equation (16)-(17) of vvh25. The theta samples can then enter in the first step
    to compute the zero quantities for the per-pulsar summaries. 
    
    Samples the theta parameters in their unconstrianed latent space via a tanh transformation
    and the Fourier coefficients are sampled via the latent coefficients y 
    which are related to the Fourier coefficients through 
    
                    a = ahat_l + L_l @ y. (subscript l for latent)
    
    By default, ahat_l and L_l are set from the summary's regularizing quantities ahat0r, L0r.
    
    NOTE: The above latent space transformation assumes a uniform prior on theta.
    
    Returns:
    
    model: numpyro model
    init_params: init values for sampler
    hyper_pars: labels of theta samples
    
    """
    
    psl = summary.psl # get the psl likelihood obj to be transf to latent space
    
    # decentering quantities
    if (ahat_l is None) != (L_l is None):
        raise ValueError("ahat_l and L_l must be provided simultaneously.")
    if ahat_l is None:
        ahat_l = summary.ahat0r
        L_l = summary.L0r
        
    n_coeff = summary.n_coeff

    # transform likelihood from real to latent space
    logL_transformed = latent_space_transformation(psl.clogL, priordict=priordict)
    
    # extract hyperpars and their boundaries needed for latent space transf.
    hyper_pars = [p for p in logL_transformed.params if "coefficients" not in p]
    hyper_indices = jnp.array([logL_transformed.columns.index(p)for p in hyper_pars], dtype = jnp.int32)
    
    a_hyper = logL_transformed.a_bounds[hyper_indices]
    b_hyper = logL_transformed.b_bounds[hyper_indices]
    low = jnp.array([_lookup_prior(p, priordict)[0] for p in hyper_pars])
    high = jnp.array([_lookup_prior(p, priordict)[1] for p in hyper_pars])

    def model():
        # sample thetas in their low/high range
        xs_hyper_list = []
        for p, lo, hi in zip(hyper_pars, low, high):
            xs_hyper_list.append(numpyro.sample(p, dist.Uniform(lo, hi)))
        xs_hyper = jnp.array(xs_hyper_list)

        # latent hyperparameters are obtained by inversion
        y_hyper = jnp.arctanh((2.0 * xs_hyper - b_hyper - a_hyper) /
                                (b_hyper - a_hyper))
        
        # sample latent Fourier coefficients
        y = numpyro.sample("y", dist.Normal(jnp.zeros(n_coeff), jnp.ones(n_coeff)))
        
        # Fourier coefficients from decentering transf.
        numpyro.deterministic("a", ahat_l + L_l @ y)
        
        ys = jnp.concatenate([y_hyper, y])
        loglik = logL_transformed(ys, ahat_l, L_l) 
        loglik += 0.5 * jnp.dot(y, y) # undoing prior on latent space Fourier coeffs
        
        numpyro.factor("logL", loglik)

    # sets initial parameters used for numpyro model
    init_params = {p: jnp.array(0.5 * (lo + hi))
                   for p, lo, hi in zip(hyper_pars, low, high)}
    init_params["y"] = jnp.zeros(n_coeff)

    return model, init_params, hyper_pars


def run_step1_model(summaries, priordict, noisedict_list = None,
                    n_warmup=256, n_samples=1024, rng_key_val=0):

    """Runs step 1 for all pulsars included in summaries.
    
    For pulsars where theta is fixed, the sampling step is skipped.  
    For pulsars where theta is non-fixed, the numpyro model of 'make_step1_model'
    is used to generate theta samples.
    
    Finally, the result computes the zero quantities based on the theta_samples (if any)
    and populates each per-pulsar summary fully, making them ready to enter the joint second step.  
    
    """
    
    rng_key = jax.random.key(rng_key_val)

    for i, summary in enumerate(summaries):
        model, init_params, hyper_pars = make_step1_model(summary, priordict)

        print(f"[{i+1}/{len(summaries)}] Running step 1 for {summary.name}")
                
        # If no hyperparaeters exist to sample from, the sampler is skipped.
        if len(hyper_pars) == 0:
            print(f"[{i+1}/{len(summaries)}] {summary.name}. Theta fixed. Skipping sampler")
            continue

        print(f"Starting sampling for {summary.name}.")
        # we assign an independent key for each pulsar using fold_in
        rng_key_i = jax.random.fold_in(rng_key, i)
        
        kernel  = infer.NUTS(model,
                             init_strategy=infer.init_to_value(values=init_params))
        sampler = infer.MCMC(kernel,
                             num_warmup=n_warmup,
                             num_samples=n_samples,
                             progress_bar=True)
        sampler.run(rng_key_i)
        summary.samples = sampler.get_samples()
    
    print("Computing zero quantities...")
    theta_samples_list = [s.theta_samples for s in summaries]
    compute_zero_quantities(summaries, noisedict_list = noisedict_list, theta_samples_list=theta_samples_list)

def build_fourier_psr_summaries(psrs, eta0_list,
                           powerlaw, components: int = 30,
                           noisedict_list=None,
                           ecorr: bool = True,
                           Tspan=None):
    
    """ Builds and returns the PulsarFourierSummary objects for each pulsar using the
    regularizer eta0 which can either be a single dict and is then shared across all pulsars, or 
    a single dict per pulsar.
    """
    n = len(psrs)

    if isinstance(eta0_list, dict):
        eta0_list = [eta0_list] * n
    if noisedict_list is None:
        noisedict_list = [{}] * n
    else:
        noisedict_list = [nd if nd is not None else {} for nd in noisedict_list]

    powerlaw_list = powerlaw if isinstance(powerlaw, list) else [powerlaw] * n
    if len(powerlaw_list) != n:
        raise ValueError(f"powerlaw list length ({len(powerlaw_list)}) != number of pulsars ({n}).")
    
    if Tspan is None:
        Tspan = signals.getspan(psrs)

    f, df, _ = signals.fourierbasis(psrs[0], components, T=Tspan)

    # build the regularized pslmodels
    pslmodels = make_fourier_model(psrs,
        psd=[partial(pl, **eta0) for pl, eta0 in zip(powerlaw_list, eta0_list)],
        components=components, noisedict=noisedict_list,
        ecorr=ecorr, Tspan=Tspan)

    summaries = []
    for psr, psl, eta0, nd, pl in zip(psrs, pslmodels, eta0_list, noisedict_list, powerlaw_list):
        
        phi0_inv, logdet_phi0 = phi_single_pulsar(eta0, f, df, pl)

        ref_nd = nd if (nd is not None and len(nd) > 0) else psr.noisedict
        ahat0r, cf_inv = psl.conditional(ref_nd)
        sigma0r  = jsp.linalg.cho_solve((cf_inv[0], True),jnp.eye(cf_inv[0].shape[0]))
        L0r = jsp.linalg.cholesky(sigma0r, lower=True)

        summaries.append(PulsarFourierSummary(name = psr.name,
            psl  = psl, eta0 = eta0,
            phi0_inv = phi0_inv, logdet_phi0 = float(logdet_phi0),
            ahat0r = ahat0r, L0r = L0r))

    return summaries

#### Step 2 example models ####

def run_step2_SPNA(summaries, psrs, phi_func, priordict,
                   components=None, Tspan=None,
                   n_warmup=1000, n_samples=3000, rng_key=0):
    """
    Joint step-2 sampler for SPNA. Runs the SPNA sequentially if multiple pulsars
    are included in summaries.

    Returns numpyro samples of SPNA for each pulsar included in summaries.
    """
    rng_key = jax.random.key(rng_key)

    for s in summaries:
        if not s.is_ready_for_step2:
            raise ValueError(f"{s.name} is not ready for step 2.")

    if components is None:
        components = summaries[0].n_coeff // 2
    if Tspan is None:
        Tspan = signals.getspan(psrs)

    # Fourier basis (shared across pulsars)
    f, df, _ = signals.fourierbasis(psrs[0], components, T=Tspan)
    
    n = len(psrs)
    phi_func_list = phi_func if isinstance(phi_func, list) else [phi_func] * n
    
    all_samples = {}
    for i, s in enumerate(summaries):
        # this pulsar's own step-1 quantities (single dense block)
        b, TtNT, log_const0 = s.b0, s.TtNT, s.log_const0

        psr_params = {}
        rn_params = []
        for eta_key, eta_val in s.eta0.items():
            size = len(eta_val) if hasattr(eta_val, '__len__') else 1  # checks free spectrum case
            rn_name = (f"{s.name}_red_noise_{eta_key}({size})" if size > 1
                       else f"{s.name}_red_noise_{eta_key}")
            psr_params[eta_key] = rn_name
            rn_params.append((rn_name, size, _lookup_prior(rn_name, priordict)))

        phi_func_i = partial(phi_func_list[i], psr_params_list=[psr_params], f=f, df=df)

        def model(rn_params=rn_params, phi_func_i=phi_func_i, b=b,
                  TtNT=TtNT, log_const0=log_const0):
            # sample each hyperparameter from its uniform prior
            rho = {}
            for rn_name, size, rng in rn_params:
                d = dist.Uniform(*rng)
                rho[rn_name] = numpyro.sample(rn_name, d.expand([size]) if size > 1 else d)
            
            # evaluating Fourier likelihood
            numpyro.factor("logL", log_fourier_likelihood(
                rho=rho, b=b, phi_func=phi_func_i, TtNT=TtNT, log_const0=log_const0))

        # set starting values to regularizer eta0
        init_params = {rn_name: jnp.asarray(s.eta0[eta_key])
                       for eta_key, rn_name in psr_params.items()}

        # creating kernel and running sampler
        print(f"[{i+1}/{len(summaries)}] Running step 2 SPNA for {s.name}")
        rng_key_i = jax.random.fold_in(rng_key, i)  
        kernel  = infer.NUTS(model, init_strategy=infer.init_to_value(values=init_params))
        sampler = infer.MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                             progress_bar=True)
        sampler.run(rng_key_i)
        all_samples[s.name] = sampler.get_samples()

    return all_samples

def run_step2_crn(summaries, psrs, commongp, curngp, crn_components, priordict,
                  components=None, Tspan=None, n_warmup=1000, n_samples=3000, rng_key=0):
    """
    Step-2 sampler for CURN using marginalized Fourier PTA likelihood.
    
    Returns numpyro samples. 
    """
    
    rng_key = jax.random.key(rng_key)

    for s in summaries:
        if not s.is_ready_for_step2:
            raise ValueError(f"{s.name} is not ready for step 2.")

    npsr = len(psrs)
    if components is None:
        components = summaries[0].n_coeff // 2
    if Tspan is None:
        Tspan = signals.getspan(psrs)

    # reshape into (npsr, 2nfreq, 2nfreq) for batching over pulsars
    b = jnp.concatenate([s.b0 for s in summaries]).reshape(npsr, 2*components)
    TtNT = jnp.stack([s.TtNT for s in summaries]).reshape(npsr, 2*components, 2*components)
    log_const0 = sum(s.log_const0 for s in summaries)

    getN_common, getN_curn = commongp.Phi.getN, curngp.Phi.getN
    all_params = getN_common.params + getN_curn.params

    phi_func = jax.jit(functools.partial(phi_crn, crn_components=crn_components,
                                         getN_common=getN_common, getN_curn=getN_curn))

    rn_params = [(p, 1, _lookup_prior(p, priordict)) for p in all_params]

    def model():
        rho = {}
        for rn_name, size, rng in rn_params:
            d = dist.Uniform(*rng)
            rho[rn_name] = numpyro.sample(rn_name, d.expand([size]) if size > 1 else d)
            
        numpyro.factor("logL", log_fourier_likelihood_batched(
            rho=rho, b=b, phi_func=phi_func, TtNT=TtNT, log_const0=log_const0))

    # setting init_params for numpyro model
    eta0_lookup = {k: v for s in summaries for k, v in s.eta0.items()}

    init_params = {}
    for rn_name, size, (lo, hi) in rn_params:
        match = (next((v for k, v in eta0_lookup.items() if rn_name.endswith(k)), None)
                 if 'red_noise' in rn_name else None)
        if match is not None:
            init_params[rn_name] = jnp.asarray(match)
        else:
            mid = 0.5 * (lo + hi)
            init_params[rn_name] = jnp.full(size, mid) if size > 1 else jnp.array(mid)

    # creating kernel and running sampler
    kernel = infer.NUTS(model, init_strategy=infer.init_to_value(values=init_params))
    sampler = infer.MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, progress_bar=True)
    sampler.run(rng_key)
    samples = sampler.get_samples()
    sampler.print_summary()
    return samples


def run_step2_hd(summaries, psrs, commongp, hdgp, gw_components, priordict,
                 components=None, Tspan=None, n_warmup=1000, n_samples=3000, rng_key=0):
    """
    Step-2 sampler for HD using marginalized Fourier PTA likelihood.
    
    Returns numpyro samples.
    """
    
    rng_key = jax.random.key(rng_key)

    for s in summaries:
        if not s.is_ready_for_step2:
            raise ValueError(f"{s.name} is not ready for step 2.")

    npsr = len(psrs)
    if components is None:
        components = summaries[0].n_coeff // 2
    if Tspan is None:
        Tspan = signals.getspan(psrs)

    b = jnp.concatenate([s.b0 for s in summaries])
    TtNT = jsp.linalg.block_diag(*[s.TtNT for s in summaries])
    log_const0 = sum(s.log_const0 for s in summaries)
    getN_common, getN_hd = commongp.Phi.getN, hdgp.Phi.getN
    all_params = getN_common.params + getN_hd.params

    phi_func = jax.jit(functools.partial(phi_hd, rn_components=components,
                                         gw_components=gw_components,
                                         getN_common=getN_common, getN_hd=getN_hd,
                                         npsr=npsr))

    rn_params = [(p, 1, _lookup_prior(p, priordict)) for p in all_params]

    def model():
        rho = {}
        
        # sample from prior
        for rn_name, size, rng in rn_params:
            d = dist.Uniform(*rng)
            rho[rn_name] = numpyro.sample(rn_name, d.expand([size]) if size > 1 else d)

        numpyro.factor("logL", log_fourier_likelihood(rho=rho, b=b, phi_func=phi_func, 
                                    TtNT=TtNT, log_const0=log_const0))

    # setting init_params for numpyro model
    eta0_lookup = {k: v for s in summaries for k, v in s.eta0.items()}

    init_params = {}
    for rn_name, size, (lo, hi) in rn_params:
        match = (next((v for k, v in eta0_lookup.items() if rn_name.endswith(k)), None)
                 if 'red_noise' in rn_name else None)
        if match is not None:
            init_params[rn_name] = jnp.asarray(match)
        else:
            mid = 0.5 * (lo + hi)
            init_params[rn_name] = jnp.full(size, mid) if size > 1 else jnp.array(mid)
    
    # creating kernel and running sampler
    kernel = infer.NUTS(model, init_strategy=infer.init_to_value(values=init_params))
    sampler = infer.MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, progress_bar=True)
    sampler.run(rng_key)
    samples = sampler.get_samples()
    sampler.print_summary()
    return samples

#### Joint sampler ####
def run_step2_joint(summaries, psrs, commongp, priordict, 
                    curngp = None, crn_components= None,
                    globalgp = None, gw_components = None,
                    components = None, n_warmup=1000, n_samples=3000, rng_key=0):
    """
    Step-2 sampler for hyperparameters together with coefficient sampling
    using CURN decentering (for globalgp) and pulsar batching for curngp.
    """
    
    
    rng_key = jax.random.key(rng_key)
    for s in summaries:
        if not s.is_ready_for_step2:
            raise ValueError(f"{s.name} is not ready for step 2.")

    npsr = len(psrs)
    if components is None:
        components = summaries[0].n_coeff // 2
    n2_block = 2 * components
    ndim = npsr * n2_block
    getN_common = commongp.Phi.getN

    if globalgp is not None:
        
        getN_global = globalgp.Phi.getN
        all_params = getN_common.params + getN_global.params
        
        phi_func = jax.jit(functools.partial(phi_hd, rn_components=components,
                                             gw_components=gw_components,
                                             getN_common=getN_common,
                                             getN_hd=getN_global, npsr=npsr))
        
        b = jnp.concatenate([s.b0 for s in summaries])
        b_p = b.reshape(npsr, n2_block)
        TtNT = jsp.linalg.block_diag(*[s.TtNT for s in summaries])
        
        xi_shape = (ndim,)

        def model():
            
            rho = {}
            for rn_name, size, rng in rn_params:
                d = dist.Uniform(*rng)
                rho[rn_name] = numpyro.sample(rn_name, d.expand([size]) if size > 1 else d)
                
            xi = numpyro.sample("xi", dist.Normal(jnp.zeros(xi_shape), jnp.ones(xi_shape)))
            logL, a = log_jointFourierHD_dCURN(rho, xi, b, b_p, phi_func, TtNT, n2_block, npsr)
            logL = logL + 0.5 * xi.T @ xi
            
            numpyro.deterministic("a", a)
            numpyro.factor("logL", logL)

    else:
        
        getN_crn = curngp.Phi.getN
        all_params = getN_common.params + getN_crn.params
                        
        phi_func = functools.partial(phi_crn, crn_components=crn_components,
                                     getN_common=getN_common,
                                     getN_curn=getN_crn)
        
        b = jnp.concatenate([s.b0 for s in summaries]).reshape(npsr, n2_block)
        TtNT = jnp.stack([s.TtNT for s in summaries])
        xi_shape = (npsr, n2_block)

        def model():
            
            rho = {}
            for rn_name, size, rng in rn_params:
                d = dist.Uniform(*rng)
                rho[rn_name] = numpyro.sample(rn_name, d.expand([size]) if size > 1 else d)
                
            xi = numpyro.sample("xi", dist.Normal(jnp.zeros(xi_shape), jnp.ones(xi_shape)))
            logL, a = log_fourier_joint_batched(rho, xi, b, phi_func, TtNT)
            logL = logL + 0.5 * jnp.sum(xi ** 2)
            
            numpyro.deterministic("a", a)
            numpyro.factor("logL", logL)

    rn_params = [(p, 1, _lookup_prior(p, priordict)) for p in all_params]

    eta0_lookup = {k: v for s in summaries for k, v in s.eta0.items()}
    init_params = {}
    for rn_name, size, (lo, hi) in rn_params:
        match = (next((v for k, v in eta0_lookup.items() if rn_name.endswith(k)), None)
                 if 'red_noise' in rn_name else None)
        init_params[rn_name] = (jnp.asarray(match) if match is not None
                                else (jnp.full(size, 0.5*(lo+hi)) if size > 1 else jnp.array(0.5*(lo+hi))))

    rng_key, xi_key = jax.random.split(rng_key)
    init_params["xi"] = jax.random.normal(xi_key, xi_shape)
    
    kernel = infer.NUTS(model, init_strategy=infer.init_to_value(values=init_params))
    sampler = infer.MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, progress_bar=True)
    sampler.run(rng_key)
    samples = sampler.get_samples()
    sampler.print_summary()
    return samples