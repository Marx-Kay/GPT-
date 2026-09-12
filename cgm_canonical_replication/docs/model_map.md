# Paper → mathematics → code → test
Source: inputs/paper/cgm.pdf, journal pp494–530. Baseline: high-school households.

| Specification | Code | Test | Status |
|---|---|---|---|
| Eq1: CRRA, gamma10, beta.96, no bequest | cgm_parameters, cgm_solve | test_analytical | passed |
| p500: survival, 20–100 | cgm_parameters, survival.csv | test_primitives, test_simulation | table vintage unidentified |
| Eq2–4: zero-mean independent normal innovations; log permanent random walk | cgm_income_process, cgm_shocks | test_primitives, test_extended | passed |
| Eq5: pension lambda P65 for age>65 | cgm_income_process | test_primitives, test_simulation | passed |
| Eq6, p500: R=1.02+.04+.157z | cgm_returns | test_extended | passed |
| Eq7–8: saving≥0, alpha∈[0,1] | cgm_policy, cgm_solve | test_simulation, test_optimality, test_analytical | passed |
| Eq9–10: W next = saving times portfolio return | cgm_transition | test_primitives, test_simulation | passed |
| p497: age100 consumes all | cgm_solve | test_simulation | passed |
| Age20 wealth and income realization | cgm_initial | initial sensitivity | explicit assumption |
| Table2: printed cubic coefficients | cgm_parameters | test_primitives | matched to printed precision |
| Eq11/12/15/16, pp522–524 | cgm_welfare_rule | test_rule_optimality, test_welfare_convergence | passed; human capital convention disclosed |
| Eq23–27: CE loss denominator is rule CE | welfare outputs | test_mc_estimator, cgm_mc_ensemble | passed |

## Normalization (independent derivation)
Write P_t=F_t exp(nu_t), x=X/P, c=C/P, s=x-c and k=1-gamma.
Homogeneity has degree k, not degree one: V_t(X,P)=P^k v_t(x).
Before retirement G=F_(t+1)/F_t exp(u_(t+1)); y'=exp(epsilon_(t+1)).
At 65→66 and thereafter G=1 and y'=lambda: the last working permanent level freezes.
Then x'=s R^p/G+y', and
v_t(x)=max {c^k/k+beta p_t E[G^k v_(t+1)(x')]}.
The envelope is v'_t=c^(-gamma). Interior saving Euler condition:
c^(-gamma)=beta p_t E[G^(-gamma)c_next^(-gamma)R^p].
Conditional portfolio derivative (apart from positive beta,p,s):
E[G^(-gamma)c_next^(-gamma)(R-Rf)]. At alpha=0 this is ≤0; at alpha=1 it is ≥0.
At s=0 Euler inequality is c^(-gamma)≥discounted marginal return; alpha is economically undefined.
There is NO c≤s restriction.

## Timing and dollar units
W_t is financial wealth before current income. Receive Y_t, form X_t=W_t+Y_t,
choose C_t and stock fraction of s_t, realize next return and next income.
Figure3 primary wealth is W_t; cash X_t and post-consumption saving are also exported to expose p497's loose terminology.
Figure2 compares at nu=0: x = plotted dollar cash / F_age, using frozen F65 after retirement.
All level outputs are thousands of 1992 USD.

## Explicit incompleteness of the publication
The paper does not provide its 80 survival values, precise NCHS vintage, full-precision polynomial, level intercept for individual characteristics, or initial distribution.
Baseline uses printed Table2 cubic coefficients, a recovered level shift 2.700381, recovered survival, zero starting financial wealth, nu20=0, independent epsilon20.
These inputs are not claimed to be verified paper facts. Sensitivity must be reported separately.
The arithmetic-normal return has negative support. Finite quadrature matches Appendix A's positive-node approximation; continuous-normal simulation must abort on negative portfolio returns, never clip them.
Thus validation establishes the stated numerical approximation; exact unbounded-normal solvency remains a specification ambiguity.

Simulation averages independent household histories to estimate the unconditional age-specific distribution conditional on survival. They are not a single calendar-economy cross-section conditional on one shared aggregate labor shock. Eq4 splits permanent risk into aggregate and idiosyncratic components, but the baseline tables do not identify that variance split; the individual solver uses the published total variance and zero return correlation. A different common-shock aggregation convention cannot be certified from the published details.
