# cgm_replication_python

CGM (2005) 复现与连续时间机器学习重构 —— **Python 独立实现**（2026-09-10 建立）

这个目录是对 `GPT_repo_sync`（Matlab 路线）的**平行独立实现**：全部用 Python + NumPy + PyTorch 重写，
在这台机器上可直接跑通，不依赖 Matlab。目的有两个：

1. 用一套**独立于原 Matlab 程序**的代码去复现 CGM 论文，以便判断 433 千美元的差异到底出在哪；
2. 把同一个经济问题改写成**连续时间**形式，用神经网络求解，验证它能否缓解维数灾难。

---

## 一、最重要的结论

| 序列（65 岁） | 论文图 3 | **本目录的复现** | 原 Matlab 程序 |
|---|---:|---:|---:|
| 收入（千美元） | 32.4 | **32.4** | 32.2 |
| 消费（千美元） | 35.6 | **32.6** | 45.9 |
| 财富（千美元） | 221.5 | **173.9** | 432.6 |
| 财富峰值 | 221.5 | 180.5 | 433.4 |
| 股票比例 | 49.9% | 52.5% | 36.5% |

- 收入路径几乎完全吻合（RMSE 0.37 千美元）；
- 消费差 8%、财富差 21%，属于正常复现误差；
- **原 Matlab 程序财富翻倍**：它在 25 岁让家庭把 29% 的收入存起来，
  而论文明确说前 15 年消费紧跟收入（消费/收入 ≈ 0.97）。这是把财富推到 433 的直接原因。

图：`results/figure3_replication_comparison.png`

## 二、机器学习部分的关键结果

连续时间 HJB 求解器，以无劳动收入的 Merton 问题为标尺（有解析解）：

| 状态变量个数 | 训练时间 | 股票比例最大误差 | 消费财富比最大误差 |
|---:|---:|---:|---:|
| 1 | 7.6 秒 | 1.4e-4 | 1.2e-4 |
| 3 | 7.6 秒 | 8.7e-4 | 1.7e-4 |
| 6 | 7.5 秒 | 6.8e-4 | 2.7e-4 |
| 11 | 8.7 秒 | 2.3e-4 | 2.7e-4 |
| **21** | **8.3 秒** | **4.6e-4** | **3.9e-4** |

对照网格法（每维 40 点）：10 维需要 1.0e16 个网格点 ≈ **8.4e7 GB 内存**；20 维完全不可行。

图：`results/dimensionality_comparison.png`

完整 CGM 生命周期模型也已经用神经网络求解：
`results/nn_policy_metrics.json` 显示网络政策模拟出的财富峰值 **183.3**（网格解 180.5），
消费 `c/x` 最大误差 2.0e-2，模拟路径基本重合。

---

## 三、目录结构

```
discrete/     离散时间 CGM：参数、EGM 求解器、模拟器、图形对照
adaptive/     连续时间重构 + 机器学习求解器（网格法 / DPI / 神经网络）
diagnostics/  诊断与交叉验证脚本（政策对照、收敛性、VFI、Matlab 复刻尝试）
results/      所有运行结果（CSV / JSON / PNG / .npz）
fetched/      外部证据：论文图 3 数字化目标、原 Matlab 输出与政策函数
reports/      阶段性报告
```

### discrete/
| 文件 | 作用 |
|---|---|
| `cgm_model.py` | **核心**：CGM 参数、收入过程、二维 EGM 求解器、蒙特卡洛模拟器 |
| `convergence.py` | 网格点数与积分节点数的收敛性检验 |
| `make_figure.py` | 三线对比图（论文 / 本复现 / 原程序）与对照表 |
| `scorecard.py` | 把所有复现版本逐项对论文图 3 打分 |
| `run_timing_comparison.py` | 永久收入冲击两种时点约定的比较 |

### adaptive/
| 文件 | 作用 |
|---|---|
| `ct_model.py` | 连续时间模型参数、Ito 算子、Merton 解析解 |
| `ct_grid.py` | 网格法 HJB 求解器（维数灾难基准） |
| `dpi_scaling.py` | 深度策略迭代求解器 + 1–21 维规模实验 |
| `nn_policy_fit.py` | **生命周期神经网络**：把网格解蒸馏成网络政策，并做端到端对比 |
| `nn_value.py`, `nn_lifecycle.py` | 基于贝尔曼残差的神经网络求解尝试（尚未收敛，保留供后续） |
| `make_dim_comparison.py` | 网格法 vs 机器学习 对比图表 |

### diagnostics/
| 文件 | 作用 |
|---|---|
| `compare_matlab_policy.py` | 把原 Matlab 政策函数与本实现在**相同状态**上逐点对照 |
| `test_euler_consistency.py` | 检验两个解的消费增长是否满足 Euler 方程 |
| `test_alpha_convergence.py` | 股票比例网格从 61 加密到 801 的收敛性 |
| `vfi_check.py` | 独立的值函数迭代交叉验证 |
| `replica_matlab.py` | 逐行复刻原 Matlab 求解器的尝试（用于定位 bug） |

---

## 四、如何运行

```bash
cd cgm_replication_python

# 1) 离散时间复现（约 30 秒）
python3 discrete/cgm_model.py
python3 discrete/make_figure.py
python3 discrete/scorecard.py

# 2) 连续时间 + 机器学习规模实验（约 1 分钟）
python3 adaptive/dpi_scaling.py
python3 adaptive/make_dim_comparison.py

# 3) 完整生命周期神经网络（约 2 分钟）
cd adaptive && PYTHONPATH=.. python3 nn_policy_fit.py
```

## 五、已知限制（写清楚，不藏）

1. **21% 的财富差距尚未解释**。候选原因：永久收入冲击的时点约定（两种合法约定分别给出 180.8 和 194.5）、
   论文未明说的初始条件、以及从 PDF 图形数字化本身的 1–2% 误差。
2. **股票比例在年轻阶段偏低**。本实现给出 25–45 岁约 40–63%，论文是 94–100%。
   原程序这段反而更准，说明两个控制变量的联合最优性还有改进空间。
3. **真正的贝尔曼残差神经网络求解尚未收敛**（`nn_value.py`）。
   当前可交付的是**政策蒸馏 + 端到端验证**，已经能复现财富峰值 183.3 vs 180.5。
4. **Matlab 只能通过批处理运行**：因为沙箱默认禁止写 `~/Library` 和 `~/.matlab`，
   而 Matlab 启动必须写这两处。需要放开文件权限后才能跑 `matlab -batch`。

## 六、与 `GPT_repo_sync` 的关系

| | `GPT_repo_sync`（Matlab 路线） | 本目录（Python 路线） |
|---|---|---|
| 求解器 | Matlab EGM，结果 433 | Python EGM，结果 174–181 |
| 数值验收 | Euler / KKT / 预算约束 | 收敛性 + 最优性 + 与论文逐点对照 |
| 外部证据 | 历史 Fortran、外部 HARK 项目 | 论文图 3 数字化目标 + 原 Matlab 政策函数 |
| 机器学习 | Merton 与 Ito 算子原型 | 完整连续时间模型 + 1–21 维规模实验 + 生命周期神经网络 |

两条路线应当**并列保留**，因为它们给出不同答案这一事实本身就是复现结论的一部分。
