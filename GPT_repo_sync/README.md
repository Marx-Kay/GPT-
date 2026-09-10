# CGM (2005) 复现与连续时间机器学习重构

本仓库保存一项正在进行的博士阶段研究：先把 Cocco、Gomes 与 Maenhout（2005）的离散时间生命周期消费—投资模型从 Francisco Gomes 的通用 Matlab/Fortran 模板适配出来，再用 Duarte、Duarte 与 Silva（2024）的连续时间机器学习方法重构。

当前状态需要明确区分：

- CGM 年度模型已经有可复跑的 Matlab 实现、三组模拟、独立 EGM 交叉验证和自动数值验收。
- 数值实现通过了预算、退休接口、终期条件、Euler/KKT 和解析特例检查。
- 论文图 2A 接近；论文图 3 和表 6 尚未完成定量复现。当前正式永久收入模型的财富峰值约 433 千美元，而论文图 3 约 222 千美元。
- RFS 方法已经完成 Ito 算子和无收入 Merton 小模型验证；真正的 CGM 生命周期神经网络、高维性能比较和“缓解维数灾难”的实证尚未完成。

因此，仓库中的报告保留了误差、假设和未解决问题，不把“程序运行成功”写成“论文全部复现”。

## 快速运行

在 MATLAB 中：

```matlab
addpath('cgm_replication/matlab_final')
run_cgm
```

主入口会求解 CGM 基准、模拟三个随机种子、执行数值验收并生成图表。临时 `.mat` 工作空间和日志被 `.gitignore` 排除；完整运行结果中的轻量 CSV、JSON、PNG 和 PDF 已保留。

在 Python 3.11 + PyTorch 中，可以运行：

```bash
python3 cgm_replication/ml/verify_ito.py
python3 cgm_replication/ml/merton_dpi.py
python3 cgm_replication/ml/verify_lifecycle.py
```

这些脚本验证 RFS 算子和小模型，不是已经完成的生命周期 ML 求解器。

## 目录

- `cgm_replication/matlab_final/`：当前主 Matlab 实现和轻量运行结果。
- `MATLAB_REPORT.md`：模型、校准、数值验收、出版图差异和后续问题。
- `cgm_replication/reports/NORMALIZATION_AUDIT.md`：永久收入归一化和历史代码模型差异审计。
- `cgm_replication/reports/continuous_time_design.md`：连续时间状态过程、HJB、退休接口和 DPI 验收方案。
- `cgm_replication/paper_targets/`：从本地论文 PDF 矢量图提取的对照目标及元数据。
- `cgm_replication/ml/`：RFS Ito 算子、Merton 小模型和连续时间局部检查。
- `cgm_replication/template_sources/`：原始 Matlab/Fortran 模板的只读副本。
- `cgm_replication/source_audit/`：源文件哈希和外部历史代码溯源证据。

原始论文 PDF、运行工作空间和 Gemini 生成的 `cgm_ml_replication` 目录没有提交到本仓库。
