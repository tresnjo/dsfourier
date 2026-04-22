Implementation of step 1 and step 2 in paper by [Valtolina et al. (2025)](http://dx.doi.org/10.1103/s3gy-km61)
using [Discovery (Vallisneri et al., 2025)](https://github.com/nanograv/discovery), and flow-correction.



Under `flow/`, there are currently three files:



1. `sp_flow.ipynb` which implements the flow-corrected posterior under fixed white noise and RN hyperparameters
2. `sp_flow_wn.ipynb` which implements flow-corrected posterior marginalizing over white noise but fixed RN hyperparameters
3. `sp_flow_eta.ipynb` which is first attempt of implementing flow-corrected with varying RN hyperparameters

