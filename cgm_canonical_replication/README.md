# CGM (2005) Canonical Discrete-Time Replication

本目录是唯一的 CGM 离散时间主工程。以论文高中文凭家庭的基准模型为规范：年度 CRRA 消费与投资、永久收入随机游走、独立暂时收入风险、65岁最后工作、66岁起退休、不借款、不卖空、无遗赠。

## 运行

在 MATLAB 中将当前目录切换到本目录，执行：

```matlab
run_all
```

或在本目录的终端执行：

```bash
matlab -batch "run_all"
```

已使用 MATLAB R2026a 验证。仅需基础 MATLAB（含默认 JVM）；无 Python、额外工具箱、网络、旧 MAT 文件或项目外路径依赖。结果全部由源码生成，运行失败会抛出 assert 并在 `results/validation/run_status.json` 留下状态。首次/再次运行方式相同。

## 参数与规范

- `src/cgm_parameters.m`：唯一经济参数入口。直接采用论文 Table2 刊载精度的三次多项式。
- `src/numerical_config.m`：冻结的网格、积分、种子和验收阈值。
- `docs/model_map.md`：论文方程→代码→测试，包含归一化推导。
- `docs/calibration.md`：逐参数对照；未刊出的收入水平截距、生存表版本和初始条件单独标为实施假设。
- `inputs/provenance/manifest.json`：PDF、原始矢量坐标和辅助输入的来源与 SHA256；每次运行自动校验。

不得将来自历史模板的未刊载数值说成已从论文完全核实。模型参数不会为了匹配图表而改变。

## 测试与结果

`run_all` 依次执行解析检验、EGM 求解、独立 Euler/KKT、网格与积分收敛、共享转移核的模拟一致性、Figure2/3、五条 Table6 基准规则、Bellman 与独立重要性抽样 MC、Figure11 和明确标记的敏感性检查。

最先看 **[本次运行汇总](results/SUMMARY.md)**，机器可读结果如下：

| 内容 | 文件 |
|---|---|
| 全部科学验收 | `results/validation/dashboard.json` |
| 本次运行与完成状态 | `results/validation/run_status.json` |
| Figure2 九条政策曲线误差 | `results/figure2/errors.csv` |
| Figure2 每个选定状态对照 | `results/figure2/selected_states.csv` |
| Figure3 收入、消费、金融财富、股票比例 | `results/figure3/lifecycle.csv` |
| Figure3 论文定量对照 | `results/figure3/comparison.csv` |
| Table6 与 MC/SE | `results/table6/welfare.csv` |
| MC 三种子与尾部诊断 | `results/table6/mc_seeds.csv` |
| Figure11 路径对照 | `results/table6/figure11_errors.csv` |
| 运行时间 | `results/validation/runtime.json` |
| 假设敏感性；不改变 baseline | `results/validation/assumption_sensitivity.csv` |

Table6 在此指**基准校准行的全部五条规则**，不声称完成论文其余异质性扩展行。其消费重新优化和福利公式见 `docs/welfare_method.md`。

## 如何解释结果

数值验收通过与出版结果匹配是两项不同结论。当前大部分 Figure2、Figure3 财富与 Table6 福利仍未匹配；不因数值更接近某个旧结果而判定正确。

具体未解决事项见 `docs/unresolved_discrepancies.md`，包括未刊载校准、初始状态、财富时点、Figure2 条件状态、部分 heuristic 实施口径，以及正态收益负尾与非负财富限制之间的规范问题。这里没有为追求论文数字进行调参。

所有核心代码均为 MATLAB。`diagnostics/` 只存参考与诊断说明。此工程不包含连续时间、HJB 或机器学习工作。

注：100岁终期全部消费、储蓄为0，文件中的股票比例0是“没有投资”的存储约定，不是一个有经济意义的资产配置选择。论文股票份额比较在终期之前进行。模拟均值是独立收入/收益历史上的期望，不是一条给定共同宏观冲击下的横截面。
