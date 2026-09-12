# Canonical discrete CGM run

Clean source run completed. Publication matching is a separate outcome.

- Two-period error: 1.11e-16
- Deterministic error: 1.07e-14
- Euler median / p95 / maximum: 1.29848e-07 / 2.80669e-06 / 0.00128047
- KKT maximum: 9.30798e-05
- Grid consumption/share change: 2.68868e-05 / 0.0029018
- Quadrature consumption/share change: 0.0010112 / 0.000112865
- Welfare refinement maximum: 4.33167e-06 percentage points
- Budget error: 9.09495e-13
- Runtime: 377.96 seconds (solver 20.72; simulation 0.79; welfare 164.33)

## Figure 2

|Panel|Age|RMSE|Max error|
|---|---:|---:|---:|
|A|99|0.024296|0.134873|
|B|20|0.279305|0.550375|
|B|30|0.287641|0.44283|
|B|55|0.168596|0.300903|
|B|75|0.0263975|0.0555353|
|C|20|9.98429|11.2705|
|C|35|8.78538|9.7388|
|C|65|1.77414|2.08286|
|C|85|1.84738|2.05941|

## Figure 3

|Statistic|Paper|Canonical|Difference|
|---|---:|---:|---:|
|wealth_peak|221.476|439.054|217.578|
|wealth_peak_age|65|66|1|
|wealth_age50|125.136|335.011|209.875|
|wealth_age65|221.476|437.024|215.548|
|income_age50|33.5718|35.2353|1.66355|
|income_age65|32.3592|35.9966|3.63743|
|consumption_peak|35.5532|48.922|13.3689|
|alpha_age65|0.499211|0.38192|-0.11729|

## Table 6 baseline row

Losses and SE are percentage points; Appendix C Eq27 denominator is rule CE.

|Rule|Paper|Bellman|MC|MC SE|MC−Bellman|
|---|---:|---:|---:|---:|---:|
|100_age|0.637|0.991879|0.992914|0.00151307|0.00103479|
|no_income|1.531|2.32855|2.33199|0.00375373|0.0034368|
|no_income_risk|0.152|0.466452|0.465313|0.00222898|-0.00113936|
|zero|2.108|5.66689|5.67337|0.00484325|0.00648345|
|approximation|0.084|4.61779|4.62077|0.00393492|0.00298025|

## Remaining limitations

Survival vintage and log income level shift are recovered assumptions; initial distribution and Figure2 conditioning are unprinted. Arithmetic-normal negative-return support conflicts with almost-sure nonnegative wealth; validation covers finite positive quadrature and realized continuous paths. Publication wealth/policies/welfare remain unmatched. See docs/unresolved_discrepancies.md.
