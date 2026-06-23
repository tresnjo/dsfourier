# Implementation of the flow-corrected Fourier-domain PTA likelihood.
# In this module, we replace the Gaussian approximation in
# the vanilla Fourier-domain PTA likelihood of (S. Valtolina, R. van Haasteren, 2025)
# using a normalizing flow. 
# Source: https://journals.aps.org/prd/abstract/10.1103/s3gy-km61
# In what follows, we refer to the aforementioned paper as 'vvh25'

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

from flowjax.train import fit_to_data
from flowjax.flows import masked_autoregressive_flow, TriangularAffine
from flowjax.distributions import StandardNormal, Transformed
import optax

from .. import (likelihood, prior, signals, matrix)
from .. import partial

from .nanograv_single_pulsar_outlier import _lookup_prior
from .fourierpta import (make_step1_model, latent_space_transformation,
                         log_fourier_joint_batched, log_jointFourierHD_dCURN,
                         log_fourier_likelihood, make_fourier_model)
from .spectral_covs import *
import discovery.flow as dsf

from .. import priordict_standard
newdict = {'(.*_)?red_noise_coefficients\\(([0-9]*)\\)': [-100, 100],
           '(.*_)?log10k': [-9.0, -4.0]}

priordict_standard.update(newdict) # for flat-tail powerlaw spectrum

def get_rn_slice(psl):
        """Picks out Fourier coefficients from full coefficient vector"""
        rn_key = [k for k in psl.N.index.keys() if 'red_noise' in k][0]
        return psl.N.index[rn_key]

@dataclasses.dataclass
class FlowPulsarFourierSummary:
    """
    Per-pulsar containers for the flow-corrected Fourer-domain based PTA analysis,
    with each pulsar object represented as an instance of the class.
    
    Step 1 involves the per-pulsar analysis, where regularized quantities
    are stored evaluated at a reference hyperpameter eta0. All regularizing
    quantities are then stored. 
    
    Whenever all necesary fields have been populated, the per-pulsar summary objects
    are ready to enter the joint analysis.
    """
    
    name: str
    psl: object # discovery PulsarLikelihood object
    eta0: dict  # hyperparameter regularizer (will be set by MAP estimate)
    
    phi0_inv: jnp.ndarray # spectral prior cov. matrix evaluated at eta0
    logdet_phi0: float  # log determinant of phi_0

    # PTA conditional quantiites evalauted with regularizer (r)
    ahat0r: jnp.ndarray # conditional mean of Fourier coeff. at eta0
    L0r: jnp.ndarray # cholesky of conditional cov. of Fourier coeff. at eta0
    log_const0: Optional[float] = 0.0 # any constant offsets (will not affect sampler hence optional)
    
    # flow-related objects
    flow: object = None # container for flow
    ahat_f: jnp.ndarray = None  # mean estimate of Fourier coefficients from flow
    Sigma_f: jnp.ndarray = None  # cov. estimate of Fourier coefficients from flow
    L_f: jnp.ndarray = None  # cholesky of Sigma_f
    bf: jnp.ndarray = None  # = Sigma_f^{-1} \ahat_f 
    TtNTf: jnp.ndarray = None # = Sigma_f^{-1} - Phi0^{-1}

    # container for numpyro samples as outputed of sampler in step 1 used for MLE estimate
    # for bijection parameters in normalizing flow
    samples: dict = None  # can also be set manually which updates theta_samples (see below)

    @property
    def n_coeff(self):
        """Number of Fourier coefficients per pulsar"""
        return self.ahat0r.shape[0]

    @property
    def rn_slice(self):
        return get_rn_slice(self.psl)

    @property
    def is_ready_for_step2(self):
        """Checks if per-pulsar summaries are ready to enter the second step
        NOTE: log_const0 is not a required quantity since it does not affect final result 
        """
        required = {'ahat0r': self.ahat0r, 'L0r': self.L0r, 'bf': self.bf, 
                    'TtNTf': self.TtNTf, 'flow': self.flow, 'ahat_f': self.ahat_f,
                    'Sigma_f': self.Sigma_f, 'L_f': self.L_f}
        
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


def extract_y_samples(summary):
    """Extracts the latent space variables y which are related to the Fourier coefficients a
    using a = L0 y + ahat0 with the decentering based on the regularized quantities ahat0r and L0r.
    
    In case y_samples are already present in the samples from a seperate analysis, they will be extracted.
    
    """
    if summary.samples is None:
        raise ValueError(f"Pulsar {summary.name!r} has missing samples.")
 
    if "y" in summary.samples:
        return jnp.array(summary.samples["y"])
 
    a_samples = jnp.array(summary.samples["a"])    
    da = a_samples - summary.ahat0r[None, :]       
    y_samples = jax.vmap(lambda d: jax.scipy.linalg.solve_triangular(summary.L0r, d, lower=True))(da)
   
    return y_samples
 
 
def affine_flow_architecture(key, n_coeff):
    """Example affine flow that can be used for the exact case."""
    bij = TriangularAffine(loc=jnp.zeros(n_coeff), arr=jnp.eye(n_coeff)) 
    return Transformed(StandardNormal((n_coeff,)), bij)
    
    
def fit_flow_to_samples(summary, rng_key, flow_architecture,
                        learning_rate = 1e-3, max_epochs = 1000,
                        batch_size = 256):

    """
    Fits a normalizing flow to the samples using FLOWJAX's fit_to_data. 
    Equivalent to MLE approach for normalizing flows (see thesis doc. again). 
    
    If no flow architecture is supplied, the normalizing flow defaults to a MAF
    with 2 flow layers and a NN architecture of four layers and 32 neurons per layer. 
    Batch_size sets the number of samples to use for the stochastic gradient-descent
    to find the optimal bijection parameters of flow.
    
    """

    y_samples = extract_y_samples(summary)
    n_coeff = y_samples.shape[1]
 
    key, flow_key, train_key = jax.random.split(rng_key, 3)
 
    if flow_architecture is None:
        
        flow = masked_autoregressive_flow(flow_key,
            base_dist=StandardNormal((n_coeff,)),
            flow_layers=8, nn_width=32, nn_depth=4,
            invert=True,) # invert = True is needed for faster log_prob evaluations.
                            # it is True by default.   
    else:
        flow = flow_architecture(flow_key, n_coeff)
 
    trained_flow, _ = fit_to_data(train_key,
        dist=flow, data=y_samples,
        learning_rate=learning_rate, max_epochs=max_epochs,
        batch_size=batch_size,
        max_patience = 100,
        val_prop = 0.2, )
 
    summary.flow = trained_flow # populates the flow field in the summaries
    print(f"Finished flow-fit to pulsar {summary.name}.")
    
 
def compute_gauss_approx_to_flow(summary, rng_key, n_flow_samples = 1000000):
    
    '''
    Computes the associated Gaussian approx to the flow MCMC-style 
    assuming flow field is already populated. 
    
    MCMC samples are generated, and the associated mean and covariance is then
    estimated using the total law of expectation and covariance (ahat_f, Sigma_f).
    
    ahat_f, Sigma_f is then used to populate bf = Sigma_f^{-1} @ ahat_f
    and TtNT_f = Sigma_f^{-1} - phi_0^{-1}
    '''
  
    if summary.flow is None:
        raise ValueError(f"Missing flow to pulsar {summary.name}")
 
    y_flow = summary.flow.sample(rng_key, (n_flow_samples,))  
    a_flow = np.array(summary.ahat0r[None, :] + (summary.L0r @ y_flow.T).T)
    
    # law of total expectation and covariance and populating flow quantities
    summary.ahat_f = jnp.mean(jnp.array(a_flow), axis=0)
    summary.Sigma_f = jnp.array(np.cov(a_flow.T))
    summary.L_f = jnp.linalg.cholesky(summary.Sigma_f)
    Sigma_f_inv = jsp.linalg.cho_solve((summary.L_f, True), jnp.eye(summary.L_f.shape[0]))
    summary.bf = Sigma_f_inv @ summary.ahat_f
    summary.TtNTf = Sigma_f_inv - summary.phi0_inv
 
 
def compute_flow_summaries(summaries, rng_key=1, n_flow_samples=1000000,
                           flow_architecture=None, learning_rate=1e-3,
                           max_epochs=1000, batch_size=128):
    
    """
    Function which fits a normalizing flow and computes a gaussian approx to the flow.
    
    If a normalizing flow is not supplied in the summary files, it defaults to using the MLE approach
    for learning the bijection parameters of the normalizing flow.
    
    If a normalizing flow is already supplied, it proceeds to compute the Gaussian approx. to the flow.
    
    """

    base_key = jax.random.key(rng_key)
    n = len(summaries)
    arch_list = flow_architecture if isinstance(flow_architecture, list) else [flow_architecture] * n

    for i, (summary, arch) in enumerate(zip(summaries, arch_list)):
        print(f"[{i+1}/{n}]: Running flow step for pulsar {summary.name}")

        fit_key, approx_key = jax.random.split(jax.random.fold_in(base_key, i), 2)

        if summary.flow is None:
            fit_flow_to_samples(summary, fit_key,
                flow_architecture=arch,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                batch_size=batch_size)

        compute_gauss_approx_to_flow(summary, approx_key, n_flow_samples=n_flow_samples)
        

def fit_flow_vi(summary, rng_key,
                flow_architecture=None, num_samples=128,
                steps=1000, learning_rate=1e-3, multibatch = 1,
                optimizer = optax.adam):
    
    """
    Fits a normalizing flow using variational infernce (VI). 
    
    If no flow architecture is supplied, the normalizing flow defaults to a MAF
    with 2 flow layers and a NN architecture of four layers and 32 neurons per layer. 
    Batch_size sets the number of samples to use for the stochastic gradient-descent
    to find the optimal bijection parameters of flow.
    
    Defaults to ADAM optimizer. Can be changed to other optimizer types 
    available in the optax library if wished (see notebook where we replace it with SGD).
    
    """

    logx = latent_space_transformation(summary.psl.clogL)
    logx_partial = jax.jit(functools.partial(logx, ahat=summary.ahat0r, L=summary.L0r))

    loss = dsf.value_and_grad_ElboLoss(logx_partial, num_samples=num_samples)

    key, flow_key, train_key = jax.random.split(rng_key, 3)
    num_params = summary.n_coeff

    if flow_architecture is None:
        flow = masked_autoregressive_flow(flow_key,
            base_dist=StandardNormal((num_params,)),
            flow_layers=2, nn_width=16, nn_depth=4,
            invert=True)
    else:
        flow = flow_architecture(flow_key, num_params)

    trainer = dsf.VariationalFit(dist=flow, loss_fn=loss, multibatch=multibatch,
        learning_rate=learning_rate,
        show_progress=True,
        optimizer = optimizer(learning_rate))

    train_key, trained_flow = trainer.run(train_key, steps=steps)
    summary.flow = trained_flow
    # TODO: should also store the trainer for each pulsar
    print(f"Finished VI flow-fit for pulsar {summary.name}.")
    
def run_step1_flow(summaries, priordict,
                   flow_architecture=None, 
                   n_warmup=256, n_samples=1024, # NUTS quantities
                   rng_key_val=0, n_flow_samples=1000000, # flow samples for Gauss. approx.
                   learning_rate=1e-2, # learning rate for flow bij. parameter update
                   steps=512, # number of VI or MLE steps
                   batch_size=64, # batch size sets no. of. samples for SGD
                   optimizer = optax.adam):
    
    """
    End-to-end first step of flow-corrected result.
    
    If there are no free hyperparameters, a normalizing flow is fit using VI. Otherwise,
    a NUTS sampler is run to generaate samples and fits a flow to the Fourier coefficients.
    
    Computes the Gaussian approx. to flow for each pulsar using compute_flow_summaries.
    
    """

    rng_key = jax.random.key(rng_key_val)
    n = len(summaries)
    arch_list = flow_architecture if isinstance(flow_architecture, list) else [flow_architecture] * n

    for i, (summary, arch) in enumerate(zip(summaries, arch_list)):
        model, init_params, hyper_pars = make_step1_model(summary, priordict)

        print(f"[{i+1}/{n}] Running step 1 for {summary.name}")
        rng_key_i = jax.random.fold_in(rng_key, i)
        
        if len(hyper_pars) == 0:
            print(f"Theta fixed. Using VI.")
            fit_flow_vi(summary, rng_key_i, flow_architecture=arch,
                        num_samples=batch_size, steps=steps,
                        learning_rate=learning_rate,
                        optimizer = optimizer)
        else:
            print(f"Starting sampling for {summary.name}.")
            kernel = infer.NUTS(model, init_strategy=infer.init_to_value(values=init_params))
            sampler = infer.MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, progress_bar=True)
            sampler.run(rng_key_i)
            summary.samples = sampler.get_samples()

    compute_flow_summaries(summaries, rng_key=rng_key_val,
              n_flow_samples=n_flow_samples,
              flow_architecture=flow_architecture, learning_rate=learning_rate,
              max_epochs=steps, batch_size=batch_size)
    
    

#### Joint posteriors for flow-corrected result ####
def log_fourier_joint_batched_flow_corrected(rho, xi, bf, phi_func, TtNTf,
                                             logL_flow_list, ahat_0r, ahat_f,
                                             L_0, L_sigma_f, log_gauss_normconst):
    
    
    """
    Reuses the joint_batched log-Fourier likelihood from fourierpta.py
    except that all quantities previously labeled by subscript zero
    are reinterprerted as flow quantities instead with the additional
    non-Gaussian flow correction entering 
    
    """

    # note that the a that is returned here is not unrolled
    logL, a = log_fourier_joint_batched(rho, xi, bf, phi_func, TtNTf) # Valtolina Fourier likelihood
    numpyro.deterministic("a", a)
    
    # non-Gaussian correction (our new result)
    
    # latent space transf.
    a_diff_0 = a - ahat_0r 
    y = jax.vmap(lambda l, r: jsp.linalg.solve_triangular(l, r, lower=True))(L_0, a_diff_0)
    # eval. flow in latent space.
    log_p_flow = jnp.sum(jnp.array([logL_flow_list[i](y[i]) for i in range(len(logL_flow_list))]))
    numpyro.deterministic("y", y)

    # Gauss. approx to flow
    a_diff = a - ahat_f
    y_gauss = jax.vmap(lambda l, r: jsp.linalg.solve_triangular(l, r, lower=True))(L_sigma_f, a_diff)
    log_p_gauss = -0.5 * jnp.sum(y_gauss ** 2) + log_gauss_normconst

    numpyro.deterministic("y_gauss", y_gauss)
    
    log_flow_diff = log_p_flow - log_p_gauss
    numpyro.factor("flowCorrection", log_flow_diff) 
    numpyro.deterministic("flow_normal_ratio", jnp.exp(log_flow_diff))

    return logL


def log_fourier_joint_batched_flow_corrected_v2(rho, xi, bf, phi_func, TtNTf,
                                             logL_flow_list, ahat_0,
                                             L_0, log_gauss_normconst):
    
    
    """
    Reuses the joint_batched log-Fourier likelihood from fourierpta.py
    except that all quantities previously labeled by subscript zero
    are reinterprerted as flow quantities instead with the additional
    non-Gaussian flow correction entering 
    
    """

    # note that the a that is returned here is not unrolled
    logL, a = log_fourier_joint_batched(rho, xi, bf, phi_func, TtNTf) # Valtolina Fourier likelihood
    numpyro.deterministic("a", a)
    
    # non-Gaussian correction (our new result)
    
    # latent space transf.
    a_diff = a - ahat_0
    y = jax.vmap(lambda l, r: jsp.linalg.solve_triangular(l, r, lower=True))(L_0, a_diff)
    numpyro.deterministic("y", y)

    log_p_flow = jnp.sum(jnp.array([logL_flow_list[i](y[i])
                                     for i in range(len(logL_flow_list))]))

    log_p_gauss = -0.5 * jnp.sum(y ** 2) + log_gauss_normconst

    log_flow_diff = log_p_flow - log_p_gauss
    numpyro.factor("flowCorrection", log_flow_diff)
    numpyro.deterministic("flow_normal_ratio", jnp.exp(log_flow_diff))

    return logL

def log_fourier_joint_dCURN_flow_corrected(rho, xi, bf, bf_p, phi_func, TtNTf,
                                             logL_flow_list, ahat_0r, ahat_f,
                                             L_0, L_sigma_f, n2_block, npsr,
                                             log_gauss_normconst):
    
    
    logL, a = log_jointFourierHD_dCURN(rho, xi, bf, bf_p, phi_func, TtNTf, n2_block, npsr)
    a_tmp = a.reshape(npsr, n2_block)
    numpyro.deterministic("a", a_tmp)
    
    a_diff_0 = a_tmp - ahat_0r
    y = jax.vmap(lambda l, r: jsp.linalg.solve_triangular(l, r, lower=True))(L_0, a_diff_0)
    log_p_flow = jnp.sum(jnp.array([logL_flow_list[i](y[i]) for i in range(len(logL_flow_list))]))
    numpyro.deterministic("y", y)

    a_diff = a_tmp - ahat_f
    y_gauss = jax.vmap(lambda l, r: jsp.linalg.solve_triangular(l, r, lower=True))(L_sigma_f, a_diff)
    log_p_gauss = -0.5 * jnp.sum(y_gauss ** 2) + log_gauss_normconst
    numpyro.deterministic("y_gauss", y_gauss)

    log_flow_diff =  log_p_flow - log_p_gauss
    numpyro.factor("flowCorrection", log_flow_diff)
    numpyro.deterministic("flow_normal_ratio", jnp.exp(log_flow_diff))

    return logL


def log_fourier_joint_dCURN_flow_corrected_v2(rho, xi, bf, bf_p, phi_func, TtNTf,
                                             logL_flow_list, ahat_0,
                                             L_0, n2_block, npsr,
                                             log_gauss_normconst):
    
    
    logL, a = log_jointFourierHD_dCURN(rho, xi, bf, bf_p, phi_func, TtNTf, n2_block, npsr)
    a_tmp = a.reshape(npsr, n2_block)
    numpyro.deterministic("a", a_tmp)
    
    a_diff_0 = a_tmp - ahat_0
    y = jax.vmap(lambda l, r: jsp.linalg.solve_triangular(l, r, lower=True))(L_0, a_diff_0)
    log_p_flow = jnp.sum(jnp.array([logL_flow_list[i](y[i])
                                     for i in range(len(logL_flow_list))]))

    log_p_gauss = -0.5 * jnp.sum(y ** 2) + log_gauss_normconst

    log_flow_diff = log_p_flow - log_p_gauss
    numpyro.factor("flowCorrection", log_flow_diff)
    numpyro.deterministic("flow_normal_ratio", jnp.exp(log_flow_diff))

    return logL


def log_fourier_joint_single_flow_corrected(rho, xi, bf, phi_func, TtNTf,
                                             logL_flow_list, ahat_0r, ahat_f, L_0, L_sigma_f,
                                             log_gauss_normconst):

    
    phi_inv, logdet_phi = phi_func(rho)
    Sigma_inv = TtNTf + phi_inv
    L_sinv = jnp.linalg.cholesky(Sigma_inv)

    # uses L_sigma_inv to do the decentering and not L_sigma
    ahat = jsp.linalg.cho_solve((L_sinv, True), bf)
    a = ahat + jsp.linalg.solve_triangular(L_sinv.T, xi, lower=False)

    # see thesis document, there is a quadratic contribution and linear contribution
    quad_a = a @ Sigma_inv @ a
    linear_a = bf @ a
    log_det_L = -jnp.sum(jnp.log(jnp.diag(L_sinv)))

    logL = -0.5 * quad_a + linear_a + log_det_L - 0.5 * logdet_phi

    # non-Gaussian correction
    a_diff_0 = a - ahat_0r  # we use the regularized conditional mean and Cholesky for the latent space transf.
    y = jsp.linalg.solve_triangular(L_0, a_diff_0, lower=True)
    
    log_p_flow = logL_flow_list[0](y)

    a_diff = a - ahat_f
    y_gauss = jsp.linalg.solve_triangular(L_sigma_f, a_diff, lower=True)
    log_p_gauss = -0.5 * jnp.sum(y_gauss ** 2) + log_gauss_normconst

    numpyro.deterministic("y", y)
    numpyro.deterministic("y_gauss", y_gauss)
    
    log_flow_diff = log_p_flow - log_p_gauss
    numpyro.factor("flowCorrection", log_flow_diff)
    numpyro.deterministic("flow_normal_ratio", jnp.exp(log_flow_diff))
    return logL, a

def log_fourier_joint_single_flow_corrected_v2(rho, xi, bf, phi_func, TtNTf,
                                             logL_flow_list, ahat_0, L_0,
                                             log_gauss_normconst):

    phi_inv, logdet_phi = phi_func(rho)
    Sigma_inv = TtNTf + phi_inv
    L_sinv = jnp.linalg.cholesky(Sigma_inv)

    ahat = jsp.linalg.cho_solve((L_sinv, True), bf)
    a = ahat + jsp.linalg.solve_triangular(L_sinv.T, xi, lower=False)

    quad_a   = a @ Sigma_inv @ a
    linear_a = bf @ a
    log_det_L = -jnp.sum(jnp.log(jnp.diag(L_sinv)))

    logL = -0.5 * quad_a + linear_a + log_det_L - 0.5 * logdet_phi

    a_diff = a - ahat_0
    y = jsp.linalg.solve_triangular(L_0, a_diff, lower=True)
    numpyro.deterministic("y", y)

    log_p_flow = logL_flow_list[0](y)
    log_p_gauss = -0.5 * jnp.sum(y ** 2) + log_gauss_normconst

    log_flow_diff = log_p_flow - log_p_gauss
    numpyro.factor("flowCorrection", log_flow_diff)
    numpyro.deterministic("flow_normal_ratio", jnp.exp(log_flow_diff))

    return logL, a


def run_step2_SPNA_flow_corrected(summaries, psrs, phi_func, priordict,
                                   components=None, Tspan=None,
                                   n_warmup=1000, n_samples=3000, rng_key=0):

    rng_key = jax.random.key(rng_key)

    for s in summaries:
        if not s.is_ready_for_step2:
            raise ValueError(f"{s.name} is not ready for step 2.")

    if components is None:
        components = summaries[0].n_coeff // 2
    if Tspan is None:
        Tspan = signals.getspan(psrs)

    f, df, _ = signals.fourierbasis(psrs[0], components, T=Tspan)

    n = len(psrs)
    phi_func_list = phi_func if isinstance(phi_func, list) else [phi_func] * n

    all_samples = {}
    for i, s in enumerate(summaries):

        bf = s.bf
        TtNTf = s.TtNTf
        ahat_0r = s.ahat0r
        ahat_f = s.ahat_f
        L_0 = s.L0r
        L_sigma_f = s.L_f
        logL_flow_list = [lambda y, s=s: s.flow.log_prob(y) - jnp.sum(jnp.log(jnp.diag(s.L0r)))]
        log_const0 = s.log_const0
        xi_shape = (2*components,)
        
        log_gauss_normconst = (-0.5 * float(s.L_f.shape[0]) * jnp.log(2 * jnp.pi)
                               - jnp.sum(jnp.log(jnp.diag(s.L_f))))
        
        psr_params = {}
        rn_params = []
        for eta_key, eta_val in s.eta0.items():
            size = len(eta_val) if hasattr(eta_val, '__len__') else 1
            rn_name = (f"{s.name}_red_noise_{eta_key}({size})" if size > 1
                       else f"{s.name}_red_noise_{eta_key}")
            psr_params[eta_key] = rn_name
            rn_params.append((rn_name, size, _lookup_prior(rn_name, priordict)))

        phi_func_i = partial(phi_func_list[i], psr_params_list=[psr_params], f=f, df=df)


        def model(rn_params=rn_params, phi_func_i=phi_func_i,
                  bf=bf, TtNTf=TtNTf, log_const0=log_const0,
                  logL_flow_list=logL_flow_list, ahat_0r=ahat_0r,
                  ahat_f=ahat_f, L_0=L_0, L_sigma_f=L_sigma_f,
                  xi_shape=xi_shape,
                  log_gauss_normconst=log_gauss_normconst):

            rho = {}
            for rn_name, size, rng in rn_params:
                d = dist.Uniform(*rng)
                rho[rn_name] = numpyro.sample(rn_name, d.expand([size]) if size > 1 else d)

            xi = numpyro.sample("xi", dist.Normal(jnp.zeros(xi_shape), jnp.ones(xi_shape)))

            logL, a = log_fourier_joint_single_flow_corrected(
                rho=rho, xi=xi, bf=bf, phi_func=phi_func_i, TtNTf=TtNTf,
                logL_flow_list=logL_flow_list, ahat_0r=ahat_0r,
                ahat_f=ahat_f, L_0=L_0, L_sigma_f=L_sigma_f,
                log_gauss_normconst=log_gauss_normconst)

            numpyro.deterministic("a", a)
            numpyro.factor("logL", logL + log_const0 + 0.5 * jnp.sum(xi ** 2))
            

        init_params = {rn_name: jnp.asarray(s.eta0[eta_key])
                       for eta_key, rn_name in psr_params.items()}
        init_params["xi"] = jnp.zeros(xi_shape)

        print(f"[{i+1}/{len(summaries)}] Running step 2 SPNA flow-corrected for {s.name}")
        rng_key_i = jax.random.fold_in(rng_key, i)
        kernel  = infer.NUTS(model, init_strategy=infer.init_to_value(values=init_params))
        sampler = infer.MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, progress_bar=True)
        sampler.run(rng_key_i)
        all_samples[s.name] = sampler.get_samples()

    return all_samples



def run_step2_joint_flow_corrected(summaries, psrs, commongp, priordict,
                                    curngp=None, crn_components=None,
                                    globalgp=None, gw_components=None,
                                    components=None, n_warmup=1000, n_samples=3000, rng_key=0):

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

    ahat_0r = jnp.stack([s.ahat0r for s in summaries])
    ahat_f = jnp.stack([s.ahat_f for s in summaries])
    L_0  = jnp.stack([s.L0r for s in summaries])
    L_sigma_f  = jnp.stack([s.L_f for s in summaries])
    logL_flow_list = [lambda y, s=s: s.flow.log_prob(y) for s in summaries]


    log_det_sigma_f = jnp.sum(jnp.array([jnp.sum(jnp.log(jnp.diag(s.L_f))) for s in summaries]))
    n_total = float(sum(s.L_f.shape[0] for s in summaries))
    log_gauss_normconst = -0.5 * n_total * jnp.log(2 * jnp.pi) - log_det_sigma_f

    if globalgp is not None:

        getN_global = globalgp.Phi.getN
        all_params = getN_common.params + getN_global.params

        phi_func = jax.jit(functools.partial(phi_hd, rn_components=components,
                                             gw_components=gw_components,
                                             getN_common=getN_common,
                                             getN_hd=getN_global, npsr=npsr))

        bf   = jnp.concatenate([s.bf for s in summaries])
        bf_p = bf.reshape(npsr, n2_block)
        TtNTf = jsp.linalg.block_diag(*[s.TtNTf for s in summaries])
        xi_shape = (ndim,)

        def model():
            rho = {}
            for rn_name, size, rng in rn_params:
                d = dist.Uniform(*rng)
                rho[rn_name] = numpyro.sample(rn_name, d.expand([size]) if size > 1 else d)

            xi = numpyro.sample("xi", dist.Normal(jnp.zeros(xi_shape), jnp.ones(xi_shape)))
            logL = log_fourier_joint_dCURN_flow_corrected(rho, xi, bf, bf_p, phi_func, TtNTf,
                logL_flow_list, ahat_0r, ahat_f, L_0, L_sigma_f, n2_block, npsr,
                log_gauss_normconst)
            
            logL = logL + 0.5 * xi.T @ xi 

            numpyro.factor("logL", logL)

    else:

        getN_crn = curngp.Phi.getN
        all_params = getN_common.params + getN_crn.params

        phi_func = functools.partial(phi_crn, crn_components=crn_components,
                                     getN_common=getN_common,
                                     getN_curn=getN_crn)

        bf    = jnp.concatenate([s.bf for s in summaries]).reshape(npsr, n2_block)
        TtNTf = jnp.stack([s.TtNTf for s in summaries])
        xi_shape = (npsr, n2_block)

        def model():
            rho = {}
            for rn_name, size, rng in rn_params:
                d = dist.Uniform(*rng)
                rho[rn_name] = numpyro.sample(rn_name, d.expand([size]) if size > 1 else d)

            xi = numpyro.sample("xi", dist.Normal(jnp.zeros(xi_shape), jnp.ones(xi_shape)))
            logL = log_fourier_joint_batched_flow_corrected(
                rho, xi, bf, phi_func, TtNTf,
                logL_flow_list, ahat_0r, ahat_f, L_0, L_sigma_f, log_gauss_normconst)
            
            logL = logL + 0.5 * jnp.sum(xi ** 2) 

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


#### MAP estimate for eta0 ####
def eta_MAP(log_posterior, log10A_bounds=(-20.0, -11.0),
                      gamma_bounds=(0.0, 7.0), n_grid=10, 
                      steps=4, zoom=0.3):
    
    """
    Zooms in on the prior grid for each step using a zoom factor specified by zoom.
    Evalautes the likelihood on a n_grid x n_grid at each step.
    
    Only compatible with powerlaw model currently. 
    
    Returns MAP estimate. 
    
    """
    
    log10A_lo, log10A_hi = log10A_bounds
    gamma_lo, gamma_hi  = gamma_bounds

    for i in range(steps):
        
        log10A_grid = jnp.linspace(log10A_lo, log10A_hi, n_grid)
        gamma_grid = jnp.linspace(gamma_lo,  gamma_hi,  n_grid)

        gmesh,ampmesh = jnp.meshgrid(gamma_grid, log10A_grid, indexing='ij')
        points_gamma = gmesh.ravel()
        points_log10A = ampmesh.ravel()
        
        # evaluate log prb on grid
        log_p_vals = jax.vmap(log_posterior)(points_log10A, points_gamma)

        idx = jnp.argmax(log_p_vals)
        gamma_map = float(points_gamma[idx])
        log10A_map = float(points_log10A[idx])
        
        # define new bounds
        
        gamma_hw = 0.5 * (gamma_hi  - gamma_lo) * zoom
        log10A_hw = 0.5 * (log10A_hi - log10A_lo) * zoom
        gamma_lo = max(gamma_bounds[0], gamma_map - gamma_hw)
        gamma_hi = min(gamma_bounds[1], gamma_map + gamma_hw)
        log10A_lo = max(log10A_bounds[0], log10A_map - log10A_hw)
        log10A_hi = min(log10A_bounds[1], log10A_map + log10A_hw)

    return jnp.array([gamma_map, log10A_map])


def hessian_MAP(log_posterior, eta_0):
    # TODO: generalize for multiple params and powerlaw models
    def lp(x):
        return log_posterior(x[1], x[0])  
    H = jax.hessian(lp)(jnp.array([float(eta_0[0]), float(eta_0[1])]))
    return -np.array(H)

def estimate_regime(hessian,prior_std_log10A=(20-11)/np.sqrt(12),
                    prior_std_gamma=(7-0)/np.sqrt(12),
                    f_1 = 1.0, f_2 = 5.0):

    '''
    Compares the standard deviation at the MAP estimate with the prior,
    which results in the informativeness ratio as defined in the thesis, and from this
    it determines whether the pulsar is informative or uninformative ()
    '''
    try:
        cov_post = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return "uninformative", np.inf
    
    # TODO: informativeness ratio r_{p,i}, generalize this in case
    # of multiple hyperparameters
    var_post_gamma = cov_post[0, 0]
    var_post_log10A = cov_post[1, 1]
    
    post_std_gamma = np.sqrt(var_post_gamma)
    post_std_log10A  = np.sqrt(var_post_log10A)
    
    ratio_gamma  = post_std_gamma  / prior_std_gamma
    ratio_log10A = post_std_log10A / prior_std_log10A
    worst_ratio = max(ratio_gamma, ratio_log10A)
    
    if worst_ratio > f_2:
        regime = "uninformative"
    elif worst_ratio > f_1:
        regime = "uncertain"
    else:
        regime = "informative"
    
    return regime, worst_ratio

def compute_eta_MAP(psrs, powerlaw, priordict, eta0_prime,
                    components=30, n_grid=30, steps=5, zoom=0.4,
                    ecorr=True, Tspan = None):

    eta0_prime_list = eta0_prime if isinstance(eta0_prime, list) else [eta0_prime] * len(psrs)

    results = {}
    eta0_map_list = []

    for psr, eta0_p in zip(psrs, eta0_prime_list):
        
        # IMPORTANT NOTE: we are essentially computing the MAP
        # for an SPNA run, hence we are setting the Tspan to be psr-specific. 
        T_psr = signals.getspan([psr]) if Tspan is None else Tspan
        f, df, _ = signals.fourierbasis(psr, components, T=T_psr)

        # create a fourier model with fixed noise for simplicity
        psl_prime = make_fourier_model([psr],
            Tspan=T_psr, psd=[partial(powerlaw, **eta0_p)],
            components=components,
            noisedict=[psr.noisedict if psr.noisedict else {}],
            ecorr=ecorr)[0]

        # compute zero quantities 
        nd = psr.noisedict if psr.noisedict else {}
        ahat0, cf_inv = psl_prime.conditional(nd)
        sigma0 = jsp.linalg.cho_solve((cf_inv[0], True), jnp.eye(cf_inv[0].shape[0]))
        L0 = jsp.linalg.cholesky(sigma0, lower=True)
        Sigma0_inv = cf_inv[0] @ cf_inv[0].T
        b0 = jsp.linalg.cho_solve((L0, True), ahat0)
        
        psr_params = {eta_key: f"{psr.name}_red_noise_{eta_key}" for eta_key in eta0_p.keys()}

        # the spectral cov. matrix is set by an SPNA
        phi_func_i = partial(phi_SPNA, psr_params_list=[psr_params], f=f, df=df, powerlaw=powerlaw)

        rho_prime = {rn_name: float(eta0_p[eta_key])
                     for eta_key, rn_name in psr_params.items()}
        phi0_inv_prime, _ = phi_func_i(rho_prime)
        TtNT = Sigma0_inv - phi0_inv_prime

        def log_posterior(log10A, gamma, b0=b0, TtNT=TtNT,
                          phi_func_i=phi_func_i, psr_params=psr_params):
            return log_fourier_likelihood(rho={psr_params["log10_A"]: log10A, psr_params["gamma"]: gamma},
                b=b0, phi_func=phi_func_i, TtNT=TtNT, log_const0=0.0)

        eta_0 = eta_MAP(log_posterior, n_grid=n_grid, steps=steps, zoom=zoom)
        hessian = hessian_MAP(log_posterior, eta_0)
        regime, ratio = estimate_regime(hessian)

        print(f"\n{psr.name}:")
        print(f"eta MAP: gamma={float(eta_0[0]):.2f}, log10_A={float(eta_0[1]):.2f}")
        print(f"{regime} with ratio = {ratio:.2f}")

        if regime in ("uninformative", "uncertain"):
            lo_hi = [_lookup_prior(f"{psr.name}_red_noise_{k}", priordict)
                     for k in eta0_p.keys()]
            eta0_out = {k: float(hi)-0.01 for k, (lo, hi) in zip(eta0_p.keys(), lo_hi)}
            print(f"Defaulting to upper bound: {eta0_out}")
        else:
            eta0_out = {"log10_A": float(eta_0[1]), "gamma": float(eta_0[0]),}

        eta0_map_list.append(eta0_out)
        results[psr.name] = {"eta_0_map": eta0_out,
            "ratio": float(ratio), "regime": regime}

    return eta0_map_list, results