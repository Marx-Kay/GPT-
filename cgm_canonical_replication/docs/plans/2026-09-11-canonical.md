# Canonical discrete CGM implementation plan
Goal: a self-contained, paper-specified MATLAB lifecycle benchmark with auditable tests.
Architecture: one parameter record, one shock/transition kernel, one solver and simulator; rules are arguments. Core runtime uses base MATLAB only.
Tech stack: MATLAB, assert, tables, JSON, native figures.

1. Read pp494–501, Eq2–10, Appendix A/C; freeze calibration and document unidentified inputs.
2. Register PDF, recovered survival, extracted targets with SHA256. No external runtime paths.
3. Implement normalized transition and EGM; validate independently derived two-period, deterministic and no-income benchmarks.
4. Independently integrate Euler/KKT; refine grid and quadrature against preregistered tolerances.
5. Implement shared-kernel simulation; verify budget, moments, retirement and level/normalized identities.
6. Compare vector targets only after numerical gates.
7. Implement all five baseline Table6 heuristics with consumption reoptimization; evaluate Bellman and importance-tilted lifetime utility independently.
8. Clean run from source, export machine-readable validation and concise README. Record remaining publication ambiguity without fitting.
No ML, remote operations or new economic model variants. Diagnostic perturbations never overwrite baseline.
