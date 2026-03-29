import discovery as ds
import matplotlib.pyplot as plt
import jax as jax
import jax.numpy as jnp
import jax.scipy as jsp
import inspect

def construct_freqs(psrs, num_frequencies):
    
    T = ds.getspan(psrs)
    f = jnp.arange(1, num_frequencies + 1, dtype = jnp.float64) / T
    df = jnp.diff(jnp.concatenate((jnp.array([0]), f)))
    
    return T, f, df

def phi_sp(rho, f, df, powerlaw):
    
    phi1 = powerlaw(f, df, **rho).repeat(2) 
    phi_inv_diag = 1.0 / phi1
    phi_inv_stacked = jnp.diag(phi_inv_diag)
    
    logdet_phi0 = jnp.sum(jnp.log(phi1)) 
    
    return phi_inv_stacked, logdet_phi0


def fouriermodel(psrs, rn_components, rn_init_params, fixed_wn = True, ecorr = True, powerlaw = ds.flat_tail_powerlaw):

    Tspan = ds.getspan(psrs)
    
    expected_params = [p for p in inspect.signature(powerlaw).parameters if p not in ('f', 'df')]

    if set(rn_init_params.keys()) != set(expected_params):
        raise ValueError(
            f"{powerlaw.__name__} expects the following params {expected_params} but instead got {list(rn_init_params.keys())}"
        )
    
    if fixed_wn:
        pslmodels = [ds.PulsarLikelihood([psr.residuals,
                                        ds.makegp_timing(psr, svd=True),
                                        ds.makenoise_measurement(psr, noisedict = psr.noisedict, ecorr=ecorr),
                                        ds.makegp_fourier(psr, ds.partial(powerlaw, **rn_init_params),
                            rn_components, name='rednoise', T = Tspan)]) for psr in psrs]
        return pslmodels
    else:
        pslmodels = [ds.PulsarLikelihood([psr.residuals,
                                        ds.makegp_timing(psr, svd=True),
                                        ds.makenoise_measurement(psr, ecorr=ecorr),
                                        ds.makegp_fourier(psr, ds.partial(powerlaw, **rn_init_params),
                            rn_components, name='rednoise', T = Tspan)]) for psr in psrs]
        return pslmodels
        
        
def create_rn_keys(psrnames):
    rn_amp_keys = [f"{psr_name}_rednoise_log10_A" for psr_name in psrnames]
    rn_gamma_keys = [f"{psr_name}_rednoise_gamma" for psr_name in psrnames]
    return rn_amp_keys, rn_gamma_keys


def run_fourier_step(psrs, pslmodels, rn_components, rn_init_params, powerlaw,
                      fixed_wn=True, priordict=ds.priordict_standard, N=1000):

    
    _, f, df = construct_freqs(psrs, num_frequencies=rn_components)
    phi0_inv_single, logdet_phi0_single = phi_sp(rn_init_params, f, df, powerlaw)

    b_stacked, sigma0_inv_blocks = [], []
    quad0 = logdet_sigma0_inv = 0.0

    if fixed_wn:
        for psr_model in pslmodels:
            print(f"Computing conditional for {psr_model.name}")
            ahat0, cf_inv = psr_model.conditional({})

            sigma0_inv_psr = cf_inv[0] @ cf_inv[0].T
            logdet_sigma0_inv += 2.0 * jnp.sum(jnp.log(jnp.diag(cf_inv[0])))

            b = sigma0_inv_psr @ ahat0
            quad0 += ahat0 @ b

            sigma0_inv_blocks.append(sigma0_inv_psr)
            b_stacked.append(b)
    else:
        for psr_model in pslmodels:
            print(f"Sampling WNPs for {psr_model.name}")
            wnp_dict = ds.sample_uniform(psr_model.logL.params, priordict, N)
            
            def single_step(wnp_i):
                ahat_i, cf_inv_i = psr_model.conditional(wnp_i)
                sigma_i = jsp.linalg.cho_solve(
                    (cf_inv_i[0], True), jnp.eye(cf_inv_i[0].shape[0])
                )
                return ahat_i, sigma_i

            ahat_array, sigmas = jax.vmap(single_step)(wnp_dict)
        
            ahat0 = jnp.mean(ahat_array, axis=0)
            # using law of total covariance, see: https://en.wikipedia.org/wiki/Law_of_total_covariance
            sigma0 = jnp.mean(sigmas, axis=0) + jnp.cov(ahat_array.T, bias=False)

            chol_sigma0 = jnp.linalg.cholesky(sigma0)
            sigma0_inv_psr = jsp.linalg.cho_solve((chol_sigma0, True), jnp.eye(sigma0.shape[0]))
            logdet_sigma0_inv += -2.0 * jnp.sum(jnp.log(jnp.diag(chol_sigma0)))

            b = sigma0_inv_psr @ ahat0
            quad0 += ahat0 @ b

            sigma0_inv_blocks.append(sigma0_inv_psr)
            b_stacked.append(b)

    sigma0_inv = jsp.linalg.block_diag(*sigma0_inv_blocks)
    phi0_inv = jsp.linalg.block_diag(*([phi0_inv_single] * len(pslmodels)))
    b = jnp.concatenate(b_stacked, axis=0)
    logdet_phi0 = len(pslmodels) * logdet_phi0_single

    return b, sigma0_inv, phi0_inv, quad0, logdet_phi0, logdet_sigma0_inv

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

    logdet_phi = 2.0*jnp.sum(jnp.log(jax.vmap(jnp.diag)(phi_chol)))
    
    phi_inv_4d = jnp.einsum('fij,fg->ifjg', phi_inv_cube, jnp.eye(2*rn_components))
    phi_inv = phi_inv_4d.reshape(npsr * 2*rn_components, npsr * 2*rn_components)
    
    return phi_inv, logdet_phi


def log_fourier_likelihood(rho, b, phi_func, TNT, log_const0):

    """
    Fourier likelihood from equation (22) in https://arxiv.org/pdf/2412.11894.

    returns log L = log_const0 - 0.5 * log |Phi(rho)|
                + 0.5 * log |Sigma(rho)|
                + 0.5 * b^T Sigma(rho) b

    where Sigma(rho)^{-1} = Sigma_0^{-1}  - Phi_0^{-1} + Phi(rho)^{-1} 
                          = TNT + Phi(rho)^{-1}.

    -rho:             Dictionary of hyperparams passed to phi_func (see next).
    -phi_func:         Function which maps rho to tuple (Phi^{-1}, log|Phi|). Use
                       ds.partial to fix all args except rho.
    -b:                Vector Sigma_0^{-1} @ ahat_0 which is precomputed in step 1.
    -TNT:              Sigma_0^{-1} - Phi_0^{-1} also precomputed from step 1
    -log_const0:       Log constant precomputed from step 1 on the form
                       0.5 * (log|Sigma_0^{-1}| - ahat_0^T Sigma_0^{-1} ahat_0
                              + log|Phi_0|).
                              
    """
    
    phi_inv, logdet_phi = phi_func(rho)
    sigma_inv = TNT + phi_inv 
    
    L_sigma_inv = jnp.linalg.cholesky(sigma_inv)  
    L_sigma_b = jsp.linalg.solve_triangular(L_sigma_inv, b, lower=True) # = L_sigma @ Sigma_0^{-1} @ ahat_0
    logdet_sigma_inv = 2.0 * jnp.sum(jnp.log(jnp.diag(L_sigma_inv)))
    quad = L_sigma_b.T @ L_sigma_b # = ahat_0^T \Sigma_0^{-1}^T \Sigma \Sigma_0^{-1} ahat_0
    
    logN_diffs = - 0.5*logdet_sigma_inv + 0.5*quad
                                                               
    logdet_ratio  = - 0.5 *logdet_phi 
    
    return logN_diffs + logdet_ratio + log_const0


