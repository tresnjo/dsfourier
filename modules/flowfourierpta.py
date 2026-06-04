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

    # regularizing quantities needed for step 2
    # by default set to None and populated
    bf: jnp.ndarray = None  # = Sigma_f^{-1} \ahat_f 
    TtNTf: jnp.ndarray = None # = Sigma_f^{-1} - Phi0^{-1}
    log_const0: Optional[float] = None # any constant offsets (will not affect sampler hence optional)
    
    # flow-related objects
    flow: object = None # container for flow
    ahat_f: jnp.ndarray = None  # mean estimate of Fourier coefficients from flow
    Sigma_f: jnp.ndarray = None  # cov. estimate of Fourier coefficients from flow
    L_f: jnp.ndarray = None  # cholesky of Sigma_f

    # container for numpyro samples as outputed of sampler in step 1 used for MLE estimate
    # for bijection parameters in normalizing flow
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



# WORKFLOW IDEA NOTES:

### 1. Set eta0 from MAP estimate of an SPNA and estimate informativeness
### use this to populate all zero_quantities
### 2. When this is done, generate samples to fit a normalizing flow
### 3. The user should be able to insert their own flow architecture
### 4. The learnt flow populats the flow field, and also computes the Gaussian aprpox.
### to the flow.
### With this, all fields are ready to enter the joint sampling. 
### Samplers from fourierpta.py can be used with non-gaussian correction (use wrappers maybe?) 