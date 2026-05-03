import discovery as ds
import matplotlib.pyplot as plt

import os
import pathlib

import jax as jax
import jax.numpy as jnp
import jax.scipy as jsp

import inspect
import numpy as np
from tqdm import tqdm
import corner 

def construct_freqs(psrs, num_frequencies):
    
    T = ds.getspan(psrs)
    f = jnp.arange(1, num_frequencies + 1, dtype = jnp.float64) / T
    df = jnp.diff(jnp.concatenate((jnp.array([0]), f)))
    
    return T, f, df

def phi_sp(rho, f, df, powerlaw):
    
    if powerlaw == ds.powerlaw or powerlaw == ds.flat_tail_powerlaw: 
        phi1 = powerlaw(f, df, **rho).repeat(2) 
        
        phi_inv_diag = 1.0 / phi1
        logdet_phi0 = jnp.sum(jnp.log(phi1))  
        phi_inv_stacked = jnp.diag(phi_inv_diag)
        
        return phi_inv_stacked, logdet_phi0

    elif powerlaw == ds.freespectrum:
        # NOTE: no .repeat(2) needed
        phi1 = powerlaw(f, df, **rho)  # this already returns (2*rn_components unlike ds.powerlaw)
        
        phi_inv_diag = 1.0 / phi1
        logdet_phi0 = jnp.sum(jnp.log(phi1))  
        phi_inv_stacked = jnp.diag(phi_inv_diag)
        
        return phi_inv_stacked, logdet_phi0

def fouriermodel(psrs, rn_components, rn_init_params, fixed_wn=True, tnequad = False, ecorr=False, powerlaw=ds.flat_tail_powerlaw):

    Tspan = ds.getspan(psrs)

    expected_params = [p for p in inspect.signature(powerlaw).parameters if p not in ('f', 'df')]

    if isinstance(rn_init_params, dict):
        params_list = [rn_init_params] * len(psrs)
    elif isinstance(rn_init_params, list):
        if len(rn_init_params) != len(psrs):
            raise ValueError(
                f"rn_init_params list length ({len(rn_init_params)}) must match number of pulsars ({len(psrs)})"
            )
        params_list = rn_init_params
    else:
        params_list = [{}] * len(psrs)

    for i, params in enumerate(params_list):
        if params and set(params.keys()) != set(expected_params):
            raise ValueError(
                f"Pulsar {i}: {powerlaw.__name__} expects params {expected_params} but got {list(params.keys())}")

    if fixed_wn:
        pslmodels = [ds.PulsarLikelihood([psr.residuals,
                                          ds.makegp_timing(psr, svd=True),
                                          ds.makenoise_measurement(psr, noisedict=psr.noisedict, tnequad = tnequad, ecorr=ecorr),
                                          ds.makegp_fourier(psr, ds.partial(powerlaw, **params),
                                                            rn_components, name='red_noise', T=Tspan)])
                     for psr, params in zip(psrs, params_list)]
    else:
        pslmodels = [ds.PulsarLikelihood([psr.residuals,
                                          ds.makegp_timing(psr, svd=True),
                                          ds.makenoise_measurement(psr, ecorr=ecorr),
                                          ds.makegp_fourier(psr, ds.partial(powerlaw, **params),
                                                            rn_components, name='red_noise', T=Tspan)])
                     for psr, params in zip(psrs, params_list)]

    return pslmodels

def create_rn_keys(psrnames):
    rn_amp_keys = [f"{psr_name}_red_noise_log10_A" for psr_name in psrnames]
    rn_gamma_keys = [f"{psr_name}_red_noise_gamma" for psr_name in psrnames]
    return rn_amp_keys, rn_gamma_keys


def run_fourier_step(psrs, pslmodels, rn_components, rn_init_params, powerlaw,
                      fixed_wn=True, priordict=ds.priordict_standard, N=1000):

    _, f, df = construct_freqs(psrs, num_frequencies=rn_components)
        
    if isinstance(rn_init_params, dict):
        params_list = [rn_init_params] * len(psrs)
    elif isinstance(rn_init_params, list):
        if len(rn_init_params) != len(psrs):
            raise ValueError(
                f"rn_init_params list length ({len(rn_init_params)}) must match number of pulsars ({len(psrs)})")
        params_list = rn_init_params
    else:
        params_list = [{}] * len(psrs)

    phi0_inv_blocks, logdet_phi0 = [], 0.0
    for params in params_list:
        phi0_inv_i, logdet_phi0_i = phi_sp(params, f, df, powerlaw)
        phi0_inv_blocks.append(phi0_inv_i)
        logdet_phi0 += logdet_phi0_i

    b_stacked, sigma0_inv_blocks, ahat0s = [], [], []
    quad0 = logdet_sigma0_inv = 0.0

    if fixed_wn:
        for psr_model in pslmodels:
            ahat0, cf_inv = psr_model.conditional({})

            sigma0_inv_psr = cf_inv[0] @ cf_inv[0].T
            logdet_sigma0_inv += 2.0 * jnp.sum(jnp.log(jnp.diag(cf_inv[0])))

            b = sigma0_inv_psr @ ahat0
            quad0 += ahat0 @ b

            sigma0_inv_blocks.append(sigma0_inv_psr)
            b_stacked.append(b)
            ahat0s.append(ahat0)
    else:
        for psr_model in pslmodels:
            ahat_list, sigmas = [], []
            for _ in tqdm(range(N), desc=f"Sampling wnps for {psr_model.name}"):
                wnp_i = ds.sample_uniform(psr_model.logL.params, priordict)
                ahat_i, cf_inv_i = psr_model.conditional(wnp_i)
                ahat_list.append(ahat_i)
                cf_inv_i = cf_inv_i[0]
                
                sigma_inv = cf_inv_i @ cf_inv_i.T
                sigma = jnp.linalg.inv(sigma_inv)
                sigmas.append(sigma)

            ahat_array = jnp.stack(ahat_list)
            ahat0 = jnp.mean(ahat_array, axis=0)
            sigma0 = jnp.mean(jnp.stack(sigmas), axis=0) + jnp.cov(ahat_array.T, bias=False)

            chol_sigma0 = jnp.linalg.cholesky(sigma0)
            sigma0_inv_psr = jsp.linalg.cho_solve((chol_sigma0, True), jnp.eye(sigma0.shape[0]))
            logdet_sigma0_inv += -2.0 * jnp.sum(jnp.log(jnp.diag(chol_sigma0)))

            b = sigma0_inv_psr @ ahat0
            quad0 += ahat0 @ b

            sigma0_inv_blocks.append(sigma0_inv_psr)
            b_stacked.append(b)
            ahat0s.append(ahat0)

    sigma0_inv = jsp.linalg.block_diag(*sigma0_inv_blocks)
    phi0_inv = jsp.linalg.block_diag(*phi0_inv_blocks)
    b = jnp.concatenate(b_stacked, axis=0)

    return ahat0s, b, sigma0_inv, phi0_inv, quad0, logdet_phi0, logdet_sigma0_inv

def compute_zero_quantities(pslmodels, psr_noisedicts):

    # NOTE: this is really just the run_fourier_step for fixed_wn but with fewer computations
    Ls, ahat0_list = [], []
    logdet_sigma0_inv = 0.0

    for psr_model, noisedict in zip(pslmodels, psr_noisedicts):

        ahat0, cf_inv = psr_model.conditional(noisedict)

        logdet_sigma0_inv += 2.0 * jnp.sum(jnp.log(jnp.diag(cf_inv[0])))

        sigma0_psr = jsp.linalg.cho_solve((cf_inv[0], True), jnp.eye(cf_inv[0].shape[0]))
        L_sigma0 = jsp.linalg.cholesky(sigma0_psr, lower=True)

        Ls.append(L_sigma0)
        ahat0_list.append(ahat0)

    return Ls, ahat0_list, logdet_sigma0_inv
            


def phi_crn(rho, crn_components, rn_amp_keys, rn_gamma_keys,
            crn_log10A_key, crn_gamma_key, getN_common, getN_curn):

    dict_common = {k: rho[k] for k in rn_amp_keys + rn_gamma_keys}
    dict_CRN    = {crn_log10A_key: rho[crn_log10A_key], crn_gamma_key: rho[crn_gamma_key]}

    PhiN_rn  = getN_common(dict_common)  
    PhiN_crn = getN_curn(dict_CRN)       

    phi_diags = PhiN_rn.at[:, :2 * crn_components].add(PhiN_crn)  
    logdet_phi = jnp.sum(jnp.log(phi_diags))
    phi_inv = 1.0 / phi_diags

    return phi_inv, logdet_phi

def phi_hd(rho, rn_components, gw_components, rn_amp_keys,
           rn_gamma_keys, gw_log10A_key, gw_gamma_key,
           getN_common, getN_hd, npsr):

    dict_common = {k: rho[k] for k in rn_amp_keys + rn_gamma_keys}
    dict_GW    = {gw_log10A_key: rho[gw_log10A_key], gw_gamma_key: rho[gw_gamma_key]}
    
    PhiN_rn = getN_common(dict_common)           
    phi_gw  = getN_hd(dict_GW)                  

    phi_rn_cube = jax.vmap(jnp.diag)(PhiN_rn.T)  
    
    phi_gw_reshaped = phi_gw.reshape(npsr, 2*gw_components, npsr, 2*gw_components)
    phi_gw_cube = jnp.diagonal(phi_gw_reshaped, axis1=1, axis2=3).transpose(2, 0, 1)
    phi_cube = phi_rn_cube.at[:2*gw_components].add(phi_gw_cube) 

    phi_chol = jax.vmap(jnp.linalg.cholesky)(phi_cube)           
    phi_inv_cube = jax.vmap(lambda L: jsp.linalg.cho_solve((L, True), jnp.eye(npsr)))(phi_chol)                                          

    logdet_phi = 2.0 * jnp.sum(jnp.log(jax.vmap(jnp.diag)(phi_chol)))
    
    phi_inv_4d = jnp.einsum('fij,fg->ifjg', phi_inv_cube, jnp.eye(2*rn_components))
    phi_inv = phi_inv_4d.reshape(npsr * 2*rn_components, npsr * 2*rn_components)
    
    return phi_inv, logdet_phi

def extract_rn_params(psrs, log10A_default = -13.0, gamma_default = 3.5):
    '''
    Extracts the IRN parameters from the noisedict.
    If absent, IRN set to (log10A, gamma) = (-13, 3.5) as default.
    '''
    rn_params = []
    for psr in psrs:
        name = psr.name
        if any('red_noise' in key for key in psr.noisedict):
            if psr.noisedict[f'{name}_red_noise_gamma'] < 0:
                gamma = -psr.noisedict[f'{name}_red_noise_gamma']
            else:
                gamma = psr.noisedict[f'{name}_red_noise_gamma']
            rn_params.append({
                'log10_A': psr.noisedict[f'{name}_red_noise_log10_A'],
                'gamma':   gamma,
            })
        else:
            rn_params.append({
                'log10_A': log10A_default,
                'gamma':   gamma_default,})
    return rn_params


def log_fourier_likelihood(rho, b, phi_func, TNT, log_const0):

    """
    Fourier likelihood corresponding to step 2 of Valtolina paper.

    returns log L = log_const0 - 0.5 * log|Phi(rho)|
                + 0.5 * log|Sigma(rho)|
                + 0.5 * b^T Sigma(rho) b

    with Sigma(rho)^{-1} = TNT + Phi(rho)^{-1}.

    :rho:              Dict of hyperparameters passed to phi_func.
    :b:                Vector Sigma_0^{-1} @ ahat_0 from step 1.
    :phi_func:         Function mapping rho to tuple (\Phi^{-1}, log|Phi|). Use
                       ds.partial to fix all args except rho.
    :TNT:              Sigma_0^{-1} - Phi_0^{-1} from step 1
    :log_const0:       Log constant from step 1 
                       0.5 * (log|Sigma_0^{-1}| - ahat_0^T Sigma_0^{-1} ahat_0
                              + log|Phi_0|).
                              
    """
    
    phi_inv, logdet_phi = phi_func(rho)
    sigma_inv = TNT + phi_inv 
    
    L_sigma_inv = jnp.linalg.cholesky(sigma_inv)  
    L_sigma_b = jsp.linalg.solve_triangular(L_sigma_inv, b, lower=True)
    logdet_sigma_inv = 2.0 * jnp.sum(jnp.log(jnp.diag(L_sigma_inv)))
    quad = L_sigma_b.T @ L_sigma_b
    
    logN_diffs = - 0.5 * logdet_sigma_inv + 0.5 * quad
                                                               
    logdet_ratio  = - 0.5 *  logdet_phi 
    
    return logN_diffs + logdet_ratio + log_const0

### NEW - for finding optimal eta
def make_marginalized_log_posterior(TtNT, b_0, phi_crn_partial, rn_amp_keys, rn_gamma_keys,
                       crn_log10A_key, crn_gamma_key, n_crn_grid=10):
    
    # returns the CRN marginalized log posterior for RN hyperparameters 
    # (works for single pulsar only as of currently)
    # marginalization is performed numerically using grid
    crn_log10A_grid = jnp.linspace(-20.0, -11.0, n_crn_grid)
    crn_gamma_grid  = jnp.linspace(0.0, 7.0, n_crn_grid)
    crna_mesh, crng_mesh = jnp.meshgrid(crn_log10A_grid, crn_gamma_grid, indexing='ij')
    crn_points = jnp.stack([crna_mesh.ravel(), crng_mesh.ravel()], axis=1)

    def log_posterior(irn_log10A, irn_gamma):

        def log_p_at_crn(crn_params):
            crn_log10A, crn_gamma = crn_params[0], crn_params[1]
            etas = {}
            for k in rn_amp_keys:
                etas[k] = irn_log10A
            for k in rn_gamma_keys:
                etas[k] = irn_gamma
            etas[crn_log10A_key] = crn_log10A
            etas[crn_gamma_key] = crn_gamma

            phi_inv_diags, logdet_phi = phi_crn_partial(etas)
            phi_inv_diag = phi_inv_diags[0] # NOTE: assumes single pulsar

            Sigma_inv = TtNT + jnp.diag(phi_inv_diag)
            L_inv = jnp.linalg.cholesky(Sigma_inv)
            log_det_Sigma = -2.0 * jnp.sum(jnp.log(jnp.diag(L_inv)))
            mu = jsp.linalg.cho_solve((L_inv, True), b_0)
            quad_b = b_0 @ mu

            return 0.5*quad_b + 0.5*log_det_Sigma - 0.5*logdet_phi

        log_p_crn = jax.lax.map(log_p_at_crn, crn_points)
        return jax.scipy.special.logsumexp(log_p_crn)

    return log_posterior


def eta_MAP(log_posterior, log10A_bounds=(-20.0, -11.0),
                      gamma_bounds=(0.0, 7.0), n_grid=10, 
                      steps=4, zoom=0.3):
    
    log10A_lo, log10A_hi = log10A_bounds
    gamma_lo, gamma_hi  = gamma_bounds

    for i in range(steps):
        log10A_grid = jnp.linspace(log10A_lo, log10A_hi, n_grid)
        gamma_grid = jnp.linspace(gamma_lo,  gamma_hi,  n_grid)

        gmesh,ampmesh = jnp.meshgrid(gamma_grid, log10A_grid, indexing='ij')
        points_gamma = gmesh.ravel()
        points_log10A = ampmesh.ravel()

        log_p_vals = jax.vmap(log_posterior)(points_log10A, points_gamma)

        idx = jnp.argmax(log_p_vals)
        gamma_map = float(points_gamma[idx])
        log10A_map = float(points_log10A[idx])
        
        gamma_range = (gamma_hi  - gamma_lo) * zoom
        log10A_range = (log10A_hi - log10A_lo) * zoom

        gamma_lo = max(gamma_bounds[0], gamma_map - gamma_range)
        gamma_hi = min(gamma_bounds[1], gamma_map + gamma_range)
        log10A_lo = max(log10A_bounds[0], log10A_map - log10A_range)
        log10A_hi = min(log10A_bounds[1], log10A_map + log10A_range)

    return jnp.array([gamma_map, log10A_map])
