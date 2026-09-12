# Welfare rules and independent validation

Only the benchmark calibration row of Table6 is in this canonical baseline; the other rows are heterogeneity extensions, not silently represented as completed here.

| Rule | Source | Stock allocation | Reoptimized control |
|---|---|---|---|
| 100-age | Eq15 | alpha=(100-age)/100 | consumption |
| No income | Eq11 | alpha=mu/(gamma sigma_R²), capped | consumption |
| No income risk | Eq12; pp523–524 | stock dollars=min(s,m(s+H)); H survival-discounted future income | consumption, including allocation's saving dependence |
| Zero | Section5 | alpha=0 | consumption |
| Approximation | Eq16 | 1 below40, 2-.025age between40–60, .5 above60 | consumption |

Eq11 is the paper's specified heuristic, not the exact annual no-income optimum. The analytical test uses the correct discrete-return portfolio optimum separately.

H excludes current received income. It discounts future median income conditional on current permanent income by Rf and survival; future innovations are set to zero to implement 'ignoring income risk'. Equation12 labels financial wealth W ambiguously relative to Eq9 timing; the implemented invested amount is post-consumption saving. Both this convention and median versus expected future wages are disclosed limitations, not quietly resolved by Figure11. H is not used by the canonical optimal baseline.

The capped state-dependent rule can create a kink and fail global concavity. Its branch inside the shared solver scans savings globally then refines the best bracket and checks the kink. Independent direct Bellman searches validate consumption, and finer-grid tests validate its welfare. Age-only rules use the same EGM kernel as the baseline.

For M=E sum beta^t survival C^(1-gamma), define k=1-gamma. Eq27 gives
L=100[(M_opt/M_rule)^(1/k)-1]. The denominator is CE under the rule.
Bellman evaluation interpolates consumption-equivalent value and uses the exact borrowing-boundary formula, not a linear value extrapolation from zero.

## MC estimator
Each policy is simulated under a predictable adaptive Gaussian proposal for permanent and return shocks, using common underlying normals and antithetic pairs. For standard normal innovation z drawn as m+e, the density ratio is exp(-m e-m²/2). Prefix log weights accumulate these exact corrections. Permanent income powers are tracked together with the weights, avoiding underflow and uncontrolled original-measure tails. Proposals depend on income/financial exposure but do not change the target law.

Current temporary income is integrated conditionally with 21 Gaussian nodes; past temporary shocks that determine wealth are independently sampled from the original distribution. This is an independent lifetime-utility path evaluator, not a fit to Bellman values. It uses no value-function labels. A consume-all-income benchmark with exact lognormal lifetime moments validates the likelihood algebra.

Three fixed independent seeds each simulate 100000 paths. Pair-level covariance produces delta-method welfare SE including covariance with the optimal rule. Seed-level results, effective sample sizes, maximum pair fractions and pooled SE are saved. Predeclared acceptance is |MC−Bellman|≤4SE+.03 percentage points, maximum pooled SE<.02 pp, and no pair supplies1% of the sampled moment. No tolerances are loosened to accommodate noise.
