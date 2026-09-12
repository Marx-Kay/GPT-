# Third-party reference, not model authority
Reviewed 2026-09-11: https://github.com/econ-ark/CGMPortfolio and Code/Python/CGMPortfolio.py.

The public implementation maps permanent income to a multiplicative random walk and accounts for retirement by changing permanent growth. It flags the paper's nu=1 normalization wording; its displayed policy comparison uses nu=0. Its published notebook reports higher accumulated resources and different policy curvature than the paper. It aggregates cash-on-hand (mNrm times P), whereas canonical Figure3 reports pre-income W and separately exports X and saving. Its welfare section displays the original paper's results rather than independently calculating all Table6 entries. Its .06 premium question is resolved here by CGM p500's explicit .04 premium.

This reading informs documentation checks only. No HARK output is an automated expected answer. The exact current calibration helper could not be fully retrieved during this run, so no assertion about its complete initial-distribution implementation is made.
