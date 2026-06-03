# This file contains functions that compute spectral covariance matrices and the
# associated log determinant for single pulsar models, HD and CURN.

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def phi_single_pulsar(rho, f, df, powerlaw):
    
    phi1         = powerlaw(f, df, **rho)
    phi_inv_diag = 1.0 / phi1
    logdet_phi   = jnp.sum(jnp.log(phi1))
    phi_inv      = jnp.diag(phi_inv_diag)

    return phi_inv, logdet_phi


def phi_SPNA(rho, psr_params_list, f, df, powerlaw):
  
    phi_inv_list = []
    logdet_total = 0.0

    for psr_params in psr_params_list:
        # map powerlaw param names to current rho values
        params_i  = {k: rho[v] for k, v in psr_params.items()}
        phi0_inv, logdet = phi_single_pulsar(params_i, f, df, powerlaw)
        phi_inv_list.append(phi0_inv)
        logdet_total += logdet

    phi_inv = jsp.linalg.block_diag(*phi_inv_list)
    return phi_inv, logdet_total


def phi_hd(rho, rn_components, gw_components, getN_common, getN_hd, npsr):
    
    dict_common = {k: rho[k] for k in getN_common.params}
    dict_gw = {k: rho[k] for k in getN_hd.params}

    PhiN_rn = getN_common(dict_common)
    phi_gw = getN_hd(dict_gw)

    phi_rn_cube = jax.vmap(jnp.diag)(PhiN_rn.T)
    phi_gw_reshaped = phi_gw.reshape(npsr, 2*gw_components, npsr, 2*gw_components)
    phi_gw_cube = jnp.diagonal(phi_gw_reshaped, axis1=1, axis2=3).transpose(2, 0, 1)
    phi_cube = phi_rn_cube.at[:2*gw_components].add(phi_gw_cube)

    phi_chol = jax.vmap(jnp.linalg.cholesky)(phi_cube)
    phi_inv_cube = jax.vmap(lambda L: jsp.linalg.cho_solve(
                                (L, True), jnp.eye(npsr)))(phi_chol)
    logdet_phi = 2.0 * jnp.sum(jnp.log(jax.vmap(jnp.diag)(phi_chol)))
    phi_inv_4d = jnp.einsum('fij,fg->ifjg', phi_inv_cube, jnp.eye(2*rn_components))
    phi_inv = phi_inv_4d.reshape(npsr * 2*rn_components, npsr * 2*rn_components)

    return phi_inv, logdet_phi


def phi_crn(rho, crn_components, getN_common, getN_curn):
    
    dict_common = {k: rho[k] for k in getN_common.params}
    dict_crn = {k: rho[k] for k in getN_curn.params}

    PhiN_rn = getN_common(dict_common)
    PhiN_crn = getN_curn(dict_crn)

    phi_diags = PhiN_rn.at[:, :2*crn_components].add(PhiN_crn)
    logdet_phi = jnp.sum(jnp.log(phi_diags))
    phi_inv = 1.0 / phi_diags

    return phi_inv, logdet_phi