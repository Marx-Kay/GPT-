# DIAGNOSTIC ONLY — NOT CANONICAL CGM SOLUTION

The primary run never loads old simulation results, Python solvers, Fortran output or HARK objects.

`src/cgm_sensitivity.m` runs registered initial-condition, printed-versus-recovered precision and dollar-scale checks using the same canonical kernel. Outputs are labelled diagnostics and never replace baseline settings.

Historical source differences and third-party results are evidence about possible documentation/version issues, not expected answers for tests.
