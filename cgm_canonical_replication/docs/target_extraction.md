# Publication target extraction audit
PDF SHA256 is registered in the input manifest. Raw paths for PDF pages 12,15,33 were re-extracted directly from the PDF on 2026-09-11 using PyMuPDF. There are no Bezier curves in these pages. Core MATLAB cgm_paper_targets regenerates targets from these registered raw primitives; Python is not a runtime dependency.

Figures 2 and 11 were visually checked against the original PDF. Staged fig2_targets.json used one common x-axis calibration for panels with different physical widths, giving a spurious endpoint 307.7 for a 300-thousand axis. Staged fig11_targets.json assigned the same curve to Optimal and Approximation because both have the same line width; it also omitted two rules. Neither staged file is accepted as a canonical target.

The replacement transformation fits actual vector tick marks (not text-box centers) independently in each panel. Line identities are explicit zero-based PDF path indices; Figure11 approximation uses its first 20 connected segments and excludes decorative crosses. Filled dashed paths are reduced to centers of unique vertices. Original raw objects and old staged targets are retained for provenance, not economic calibration.

Figure2 mapping: A age99 paths11:324; B age20 339:341, age30 342:343, age55 344:624, age75 625:627; C age20 965:968, age35 962:964, age65 959:961, age85 642:958. Legends were read from the original page.
Figure11 mapping: Optimal11:12; 100-age13:170; NoIncome171:249; NoIncomeRisk250; Approximation251 first20 segments. Figure3 reproduction is checked against the earlier independent vector extraction.

Targets are graph readings, not the authors simulation data. Expected reading accuracy: wealth ~2 thousand dollars, consumption/income ~1 thousand, share ~0.01. No policy parameter is inferred from these targets.
