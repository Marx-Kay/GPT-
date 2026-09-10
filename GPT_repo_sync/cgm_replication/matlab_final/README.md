# CGM 年度模型：Matlab 运行入口

这个目录实现 CGM（2005）高中教育组的年度消费—投资问题。它以提供的生命周期模板为起点完成参数、退休转移和模拟适配，并用独立的内生网格法（EGM）交叉验证。**已通过所列数值检查；尚未复现出版图 3 和表 6 的数值。**

在项目根目录打开 Matlab，运行：

```matlab
addpath(fullfile(pwd,'research_work','matlab_final'))
run_cgm
```

或先把当前目录切到本目录，直接运行 `run_cgm`。入口通过自身位置寻找输入，不依赖调用者的当前目录。运行会覆盖本目录的生成结果，不会修改原始 Matlab / Fortran 下载包。已在 MATLAB R2026a 实际运行；不依赖 Python 或额外 Matlab 工具箱。

## 一次运行做什么

1. 从 `cgm_parameters.m` 读取论文参数，求解 20–100 岁的消费和股票配置；默认 801 个储蓄网格点、每类冲击 9 个正态积分节点。
2. 分别模拟三个固定随机种子，每组 10,000 条生命周期路径。
3. 检查预算、借贷约束、退休永久收入冻结、终期消费、独立 11 节点 Euler / KKT 条件，以及两个解析特例；检查失败时终止。
4. 输出政策图、生命周期对照图和逐年龄结果。
5. 对四条固定股票配置规则重新优化消费，独立递推价值函数，计算消费等价福利损失。

参数与主求解仅从原始模板读取 80 个生存概率；没有读取任何旧求解 `.mat` 文件。绘图另外读取 `../paper_targets/` 中已从论文 PDF 提取的曲线。若存在外部历史代码诊断的结果文件，图中会额外展示该诊断曲线；主模型求解与模拟不依赖它。

## 主要文件

| 文件 | 用途 |
|---|---|
| `cgm_parameters.m`、`cgm_income.m` | 论文基准参数和收入函数 |
| `cgm_solve.m` | 年度 EGM 与股票份额条件最优化 |
| `cgm_simulate.m` | 逐家庭美元预算模拟；永久冲击累积，退休后冻结 |
| `validate_cgm.m`、`verify_analytical.m` | 独立积分节点、约束和解析验收 |
| `cgm_evaluate_policy.m`、`run_welfare.m` | 固定规则下重新优化消费及 Bellman 福利评价 |
| `check_welfare_convergence.m` | 801/1601 网格、5/9/11 求解节点，统一 11 节点评价的独立实验 |
| `build_figures.m` | 出版图对照；财富、现金和人力财富口径显式区分 |
| `run_log.txt` | 完整主入口的实际运行日志 |
| `numerical_validation.json`、`metrics.json` | 数值检查与论文匹配分别记录 |
| `figure2_policies.pdf`、`figure3_comparison.pdf` | 可放入研究报告的矢量图 |
| `welfare_table6_partial.csv` | 四条规则的计算结果与论文表 6 对照 |
| `welfare_convergence.csv` | 福利收敛实验结果 |

无表头 CSV 的列顺序：

- `cgm_seed_*.csv`：年龄、平均消费、期初金融财富、当期收入、消费后储蓄、平均股票份额、财富标准误、份额标准误。
- `seed_robustness.csv`：种子、财富峰值、峰值年龄、该年龄财富标准误、30 岁股票份额、60 岁股票份额。
- `final_lifecycle_with_se.csv`：年龄、平均消费、期初金融财富、收入、平均股票份额、财富标准误、份额标准误。
- `human_wealth_defined.csv`：年龄、平均未来收入现值、个体“未来收入现值 / 手头现金”比率的均值。
- `optimality_checks.csv`：年龄、最大/中位消费 Euler 相对误差、最大标准化股票 KKT 误差、现金恒等式误差、最低消费、最低/最高股票份额。

金额均为千 1992 年美元；份额为 0–1；福利列为百分比。标准误以 5,000 对反向冲击路径为独立观测单位，不能把 10,000 条路径当作彼此独立。

## 阅读结果时的边界

`metrics.json` 中的 `numeric_model_validated` 与 `published_figure3_reproduced` 是两件事。当前分别为 `true` 和 `false`。图 2A 的平均误差较小，不代表工作年龄政策、生命周期图或整个论文已经复现。

四条福利规则均固定投资份额后重新优化消费。初始金融财富设为零，初始暂时收入冲击按论文分布积分；论文没有在现有材料中明确给出完全相同的初始福利评价口径，因此该口径单列为假设。第五条“无收入风险”规则随内生储蓄变化，不能直接套用常数份额的 Euler 方程，当前没有用简化替代品填入该列。

主报告见 [MATLAB_REPORT.md](../../MATLAB_REPORT.md)，永久收入推导与历史代码证据见 [NORMALIZATION_AUDIT.md](../reports/NORMALIZATION_AUDIT.md)。
