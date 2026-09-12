# Frozen calibration audit
| Parameter | Paper | Canonical | Source | Match? |
|---|---:|---:|---|---|
| gamma |10|10|Table4|yes|
| beta |.96|.96|Table4|yes|
| bequest |0|0|Table4|yes|
| Rf gross |1.02|1.02|p500/Table4|yes|
| risky mean gross |1.06|1.06|p500/Table4|yes|
| equity premium |.04|.04|p500; Eq6|yes; Table4's .06 is net stock return|
| stock sigma |.157|.157|Table4|yes|
| permanent variance |.0106|.0106|Table3/4|yes|
| transitory variance |.0738|.0738|Table3/4|yes|
| shock correlations |0|0|Eq3/Table4|yes|
| final working age |65|65|Eq2/5|yes|
| terminal age |100|100|p500 and p497|yes; terminal consumption convention|
| pension ratio |.68212|.68212|Table2|yes|
| cubic coefficients |[-2.1700,.1682,-.0323,.0020]|same; age²/10 and age³/100|Table2|yes, printed precision|
| log income level shift |not printed|2.700381|recovered template|implementation assumption|
| survival sequence |NCHS; no values/vintage|registered CSV|p500 plus recovered template|cannot fully verify|
| W20 |not explicit|0|implementation assumption|sensitivity|
| nu20 |not explicit|0|implementation assumption|sensitivity|
| epsilon20 |N(0,.0738)|same|Eq2|yes|

Using printed coefficients avoids silently elevating historical full precision to paper authority. A full-precision sensitivity is diagnostic only. No mean-one adjustment is applied to lognormal shocks: the paper specifies zero log mean.
Numerical settings are exclusively in src/numerical_config.m. Configuration snapshots and hashes are written each run before solving. No comparison-driven tuning is allowed.

The exact recovered source lines and source-file SHA256 are retained in `inputs/provenance/recovered_calibration_excerpt.txt` and the manifest. Recovery documents the chosen implementation input; it does not turn an unpublished number into a published calibration fact.
