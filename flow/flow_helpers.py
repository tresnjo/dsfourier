import sys
sys.path.append('..') 

import re
import discovery as ds
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import infer
from fourierpta import *

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import discovery as ds
import discovery.flow as dsf
from flowjax.flows import masked_autoregressive_flow
from flowjax.distributions import StandardNormal
# from flowjax.train import fit_to_data (will be needed for WN marginalized result)

newdict = {'(.*_)?red_noise_coefficients\\(([0-9]*)\\)': [-100, 100]}

def simple_dict_transformation(func,priordict=newdict):
    """change from dictionary as input to list of arrays as input

    Parameters
    ----------
    func : discovery likelihood
        discovery likelihood function
    """
    priordict = {**ds.priordict_standard, **priordict}

    # figure out slices when there are vector arguments
    slices, offset = [], 0

    for par in func.params:
        if '(' in par:
            l = int(par[par.index('(')+1:par.index(')')]) if '(' in par else 1
            slices.append(slice(offset, offset+l))
            offset = offset + l
        else:
            slices.append(offset)
            offset = offset + 1
    # build vectors of DF column names and of lower and upper uniform limits
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
    a, b = ds.matrix.jnparray(a), ds.matrix.jnparray(b)

    def to_dict_and_jacobian(ys, ahat, L):
        hyper_pars = [p for p in func.params if "coefficients" not in p]
        coeff_pars = [p for p in func.params if "coefficients" in p]
        len_hyper = len(hyper_pars)

        if len_hyper > 0:
            ys_hyper = ys[-len_hyper:]
            xs_hyper = 0.5 * (b[-len_hyper:] + a[-len_hyper:] + 
                            (b[-len_hyper:] - a[-len_hyper:]) * jnp.tanh(ys_hyper))
            hyper_jacobian = jnp.sum(jnp.log(2.0) - 2.0 * jnp.logaddexp(ys_hyper, -ys_hyper))
            hyper_dict = dict(zip(hyper_pars, jnp.array(xs_hyper).T))
        else:
            hyper_jacobian = 0.0
            hyper_dict = {}

        jac = jnp.sum(jnp.log(jnp.diag(L)))
        coeff_idx = func.params.index(coeff_pars[0])
        xs_coeff = L @ ys[slices[coeff_idx]] + ahat
        hyper_dict.update({coeff_pars[0]: xs_coeff})

        return hyper_dict, jac + hyper_jacobian

        
    def transformed(ys, ahat, L):
        mydict, jac = to_dict_and_jacobian(ys, ahat, L)
        return func(mydict) + jac
    transformed.params = func.params
    transformed.to_dict_and_jacobian = to_dict_and_jacobian
    return transformed


def mcmc_rnwn(logL_test, ahat0, L0, a_hyper, b_hyper, low, high):
    
    xs_hyper = numpyro.sample("eta", dist.Uniform(low, high))
    y_hyper  = jnp.arctanh((2.0 * xs_hyper - b_hyper - a_hyper) / (b_hyper - a_hyper))

    xi = numpyro.sample("xi", dist.Normal(jnp.zeros(60), jnp.ones(60)))
    numpyro.deterministic("a", ahat0 + L0 @ xi)

    ys = jnp.concatenate([y_hyper, xi])
    loglik = logL_test(ys, ahat0, L0) + 0.5 * jnp.dot(xi, xi)
    numpyro.factor("logL", loglik)
    
    
def fit_MAF_flow(logx, ahat0, L, num_samples, num_params, rng,
                 annealing_schedule = lambda i: min(1.0, 0.5 + 0.5*i/25)):
    
    logx_partial = jax.jit(ds.partial(logx, ahat=ahat0, L=L))
    
    loss = dsf.value_and_grad_ElboLoss(logx_partial, num_samples=num_samples)
    flow_key, train_key = jax.random.split(rng, 2)
    
    flow = masked_autoregressive_flow(flow_key,base_dist=StandardNormal((num_params,)),
                                      flow_layers=2, nn_width=16, nn_depth=4,
                                      invert=True)  # using invert = True
                    # which allows for faster logprob evaluation,
                    # see https://danielward27.github.io/flowjax/api/flows.html
    
    trainer = dsf.VariationalFit(dist=flow, loss_fn=loss, multibatch=1,
                             learning_rate=1e-2, annealing_schedule=annealing_schedule,
                             show_progress=True)
    
    train_key, trained_flow = trainer.run(train_key, steps=1001)
    return train_key, trained_flow, trainer

def fit_flows(pslmodels, logxs, ahat0_list, Ls, num_samples, num_params, rng):
    
    trained_flows, train_keys = [], []
    
    for i in range(len(pslmodels)):
        print(f"Fitting flow for pulsar {pslmodels[i].name}")
        rng, subkey = jax.random.split(rng)
        train_key, trained_flow, trainer = fit_MAF_flow(
            logxs[i], ahat0_list[i], Ls[i],
            num_samples, num_params, subkey)
        trained_flows.append(trained_flow)
        train_keys.append(train_key)
        
    return trained_flows, train_keys


def gauss_approx_flow_mpsrs(trained_flows, train_keys, ahat0_list, Ls,
                             num_flow_samples=100000):

    def gauss_approx_flow(flow, ahat0, L0, train_key):
        samples_ys = flow.sample(train_key, sample_shape=(num_flow_samples,))
        a_samples = jax.vmap(lambda ys: L0 @ ys + ahat0)(samples_ys)
        
        ahat_f = jnp.mean(a_samples, axis=0)
        Sigma_f = jnp.cov(a_samples.T)
        L_f = jnp.linalg.cholesky(Sigma_f)
        
        return ahat_f, Sigma_f, L_f

    ahat_f_list, Sigma_f_list, L_f_list = [], [], []
    for flow, key, ahat0, L0 in zip(trained_flows, train_keys, ahat0_list, Ls):
        ahat_f, Sigma_f, L_f = gauss_approx_flow(flow, ahat0, L0, key)
        ahat_f_list.append(ahat_f)
        Sigma_f_list.append(Sigma_f)
        L_f_list.append(L_f)
        
    return (jnp.stack(ahat_f_list), jnp.stack(Sigma_f_list), jnp.stack(L_f_list))


def eval_flow_quantity_mpsrs(ahat_f, Sigma_f, L_f):

    def eval_flow_quantities(ahat_f_i, Sigma_f_i, L_f_i):
        
        Sigma_f_inv = jnp.linalg.inv(Sigma_f_i)
        logdet_sigma_flow_inv = -2.0 * jnp.sum(jnp.log(jnp.diag(L_f_i)))
        b_flow = Sigma_f_inv @ ahat_f_i
        quad_f = ahat_f_i @ b_flow
        return Sigma_f_inv, logdet_sigma_flow_inv, b_flow, quad_f

    results = [eval_flow_quantities(ahat_f[i], Sigma_f[i], L_f[i])
               for i in range(ahat_f.shape[0])]
    return (jnp.stack([r[0] for r in results]), jnp.stack([r[1] for r in results]),
            jnp.stack([r[2] for r in results]), jnp.stack([r[3] for r in results]))


def TtNT_mpsrs(Sigma_f_inv, params_list, f, df, powerlaw):

    def compute_TNT_flow(Sigma_f_inv_i, params):
        phi0_inv, logdet_phi0 = phi_sp(params, f, df, powerlaw)
        TNT_flow = Sigma_f_inv_i - phi0_inv
        return TNT_flow, logdet_phi0

    TNT_list, logdet_phi0_total = [], 0.0
    for i, params in enumerate(params_list):
        TNT, logdet_phi0 = compute_TNT_flow(Sigma_f_inv[i], params)
        TNT_list.append(TNT)
        logdet_phi0_total += logdet_phi0
    return jnp.stack(TNT_list), logdet_phi0_total


def make_model_crn_flow(b, phi, log_const, TNT, npsr, rn_components,
                        rn_amp_keys, rn_gamma_keys, crn_log10A_key, crn_gamma_key,
                        logL_flow_list, ahat_0, L_0, ahat_f, L_sigma_f):
    
    # checking shapes so that computations can be batched over psrs
    if TNT.shape != (npsr, 2 * rn_components, 2 * rn_components):
        TNT = TNT.reshape(npsr, 2 * rn_components, npsr, 2 * rn_components).diagonal(axis1=0, axis2=2).transpose(2, 0, 1)
    
    if b.shape != (npsr, 2 * rn_components):
        b = b.reshape(npsr, 2 * rn_components)
    
    if L_sigma_f.shape != (npsr, 2 * rn_components, 2 * rn_components):
        L_sigma_f = L_sigma_f.reshape(npsr, 2 * rn_components, npsr, 2 * rn_components).diagonal(axis1=0, axis2=2).transpose(2, 0, 1)
    
    if ahat_f.shape != (npsr, 2 * rn_components):
        ahat_f = ahat_f.reshape(npsr, 2 * rn_components)
        
    if ahat_0.shape != (npsr, 2 * rn_components):
        ahat_0 = ahat_0.reshape(npsr, 2 * rn_components)
        
    if L_0.shape != (npsr, 2 * rn_components, 2 * rn_components):
        L_0 = L_0.reshape(npsr, 2 * rn_components, npsr, 2 * rn_components).diagonal(axis1=0, axis2=2).transpose(2, 0, 1)

    def model_crn():
        etas = {}
        for k in rn_amp_keys:
            etas[k] = numpyro.sample(k, dist.Uniform(-20, -11))
        for k in rn_gamma_keys:
            etas[k] = numpyro.sample(k, dist.Uniform(0, 7))
        etas[crn_log10A_key] = numpyro.sample(crn_log10A_key, dist.Uniform(-20, -11))
        etas[crn_gamma_key]  = numpyro.sample(crn_gamma_key,  dist.Uniform(0, 7))

        xis = numpyro.sample("xi", dist.Normal(jnp.zeros((npsr, 2 * rn_components)),
                                        jnp.ones((npsr, 2 * rn_components))))

        phi_inv_diags, logdet_phi = phi(etas)
        sigma_inv = TNT + jax.vmap(jnp.diag)(phi_inv_diags)
        L_sinv = jax.vmap(jnp.linalg.cholesky)(sigma_inv)
        ahat = jax.vmap(lambda l0, bv: jsp.linalg.cho_solve((l0, True), bv))(L_sinv, b)
        Sigma = jax.vmap(lambda l0: jsp.linalg.cho_solve((l0, True), jnp.eye(2 * rn_components)))(L_sinv)
        L_sigma = jax.vmap(jnp.linalg.cholesky)(Sigma)

        a = numpyro.deterministic("a", ahat + jax.vmap(jnp.dot)(L_sigma, xis))

        quad_b = jnp.sum(jax.vmap(jnp.dot)(b, ahat))
        log_det_L = -jnp.sum(jax.vmap(lambda l0: jnp.sum(jnp.log(jnp.diag(l0))))(L_sinv))
        logL = 0.5 * quad_b - 0.5 * logdet_phi + log_const + log_det_L
        numpyro.factor("logL", logL)

        a_diff_0 = a - ahat_0
        y = jax.vmap(lambda l, r: jsp.linalg.solve_triangular(l, r, lower=True))(L_0, a_diff_0)
        numpyro.deterministic("y", y)
        log_p_flow = jnp.sum(jnp.array([logL_flow_list[i](y[i]) for i in range(npsr)]))

        a_diff = a - ahat_f
        y_gauss = jax.vmap(lambda l0, r: jsp.linalg.solve_triangular(l0, r, lower=True))(L_sigma_f, a_diff)
        log_p_gauss = -0.5 * jnp.sum(y_gauss ** 2)
        numpyro.deterministic("y_gauss", y_gauss)

        numpyro.factor("logFlow_correction", log_p_flow - log_p_gauss)

    return model_crn
