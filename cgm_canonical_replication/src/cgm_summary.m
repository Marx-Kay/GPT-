function cgm_summary(root,v,comparison,w,timing)
f=fopen(fullfile(root,'results','SUMMARY.md'),'w');clean=onCleanup(@()fclose(f));
fprintf(f,'# Canonical discrete CGM run\n\nClean source run completed. Publication matching is a separate outcome.\n\n');
fprintf(f,'- Two-period error: %.3g\n- Deterministic error: %.3g\n- Euler median / p95 / maximum: %.6g / %.6g / %.6g\n- KKT maximum: %.6g\n',v.analytical.two_period,v.analytical.deterministic,v.optimality.euler_median,v.optimality.euler_p95,v.optimality.euler_max,v.optimality.kkt_max);
fprintf(f,'- Grid consumption/share change: %.6g / %.6g\n- Quadrature consumption/share change: %.6g / %.6g\n- Welfare refinement maximum: %.6g percentage points\n- Budget error: %.6g\n',v.convergence.policy(1,:),v.convergence.policy(2,:),v.welfare_convergence.max_difference,v.simulation.budget);
fprintf(f,'- Runtime: %.2f seconds (solver %.2f; simulation %.2f; welfare %.2f)\n\n',timing.total,timing.solver,timing.simulation,timing.welfare);
fprintf(f,'## Figure 2\n\n|Panel|Age|RMSE|Max error|\n|---|---:|---:|---:|\n');
for i=1:height(comparison.figure2),t=comparison.figure2(i,:);fprintf(f,'|%s|%d|%.6g|%.6g|\n',t.panel,t.age,t.RMSE,t.max_absolute_error);end
fprintf(f,'\n## Figure 3\n\n|Statistic|Paper|Canonical|Difference|\n|---|---:|---:|---:|\n');
for i=1:height(comparison.figure3),t=comparison.figure3(i,:);fprintf(f,'|%s|%.6g|%.6g|%.6g|\n',t.statistic,t.paper,t.canonical,t.difference);end
fprintf(f,'\n## Table 6 baseline row\n\nLosses and SE are percentage points; Appendix C Eq27 denominator is rule CE.\n\n|Rule|Paper|Bellman|MC|MC SE|MC−Bellman|\n|---|---:|---:|---:|---:|---:|\n');
for i=2:height(w.table),t=w.table(i,:);fprintf(f,'|%s|%.6g|%.6g|%.6g|%.6g|%.6g|\n',t.rule,t.paper_loss,t.Bellman,t.MonteCarlo,t.MC_SE,t.MC_minus_Bellman);end
fprintf(f,'\n## Remaining limitations\n\nSurvival vintage and log income level shift are recovered assumptions; initial distribution and Figure2 conditioning are unprinted. Arithmetic-normal negative-return support conflicts with almost-sure nonnegative wealth; validation covers finite positive quadrature and realized continuous paths. Publication wealth/policies/welfare remain unmatched. See docs/unresolved_discrepancies.md.\n');
end
