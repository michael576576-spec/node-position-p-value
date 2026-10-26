# 节点中心性模拟实验

在本项目目录下运行以下命令。需要 Python 3.10 或更新版本。历史模拟与真实网络实验的
中心性指标可由 `--metric` 指定，可选 `degree`、`lac`、`eigenvector`、`betweenness`、`closeness`、`gpg`，
默认是 `degree`。这些历史实验的非度指标需要 NetworkX，绘图需要 matplotlib。

GNP 方法比较实验还需要 `igraph`、NumPy 和 SciPy：

```powershell
python -m pip install igraph numpy scipy
```

`degree` 使用原始度数；在固定节点数的图上，将其除以 `N-1` 得到的标准化
度中心性只改变尺度，不改变排序或经验 p 值。LAC 为邻居诱导子图内的平均连接度，
EC 为特征向量中心性，BC 为标准化介数中心性，CC 为不连通图采用可达比例校正的接近中心性，
GPG 为几何潜能增益，衰减系数设为 `0.5/λmax`。在不连通图上，EC 使用整图主特征向量，
非主谱分量中的节点分数可能接近零。指标计算集中在 [centrality.py](experiment_support/centrality.py)，GNM
零分布与 p 值检验集中在 [network_core.py](experiment_support/network_core.py)。新增指标时，
在 `experiment_support/centrality.py` 注册一个返回所有节点分数的函数即可。

当前直接运行的四个实验入口放在主目录：`compare_basic_model_budgets.py`、
`compare_null_models.py`、`compare_central_structure.py`、
`real_network_model_budget_comparison.py`。共用算法和历史实验入口放在
`experiment_support/`；历史入口从项目根目录使用 `python -m experiment_support.<模块名>` 运行。

## GNP 模拟图方法比较实验

```powershell
python -m experiment_support.experiment
```

该入口按 [experiment.md](experiment.md) 生成 4 组负例和 4 组正例，每组 10 张图。
负例和正例均覆盖 `N=500`、`N=1000` 与 `p=0.01`、`p=0.005`；正例中心节点占 1%
（分别为 5、10 个），其相关边的生成概率为普通节点边概率的 6 倍。
比较 DC、LAC、EC、BC、CC、GPG 六项中心性指标。
默认显著性阈值为 `0.01`；每张目标图默认使用 100 张 GNM 零图。
configuration 和 Chung–Lu 可通过 `--null-models` 选择。正负例汇总报告判定为中心节点数、
判定为非中心节点数、False Positives、Accuracy 和运行时间。

结果写入独立的 `experiment_results/benchmark_n500_n1000_six_metrics_v1/` 目录，包括逐图汇总、
按参数条件汇总和 settings 配置记录。添加 `--save-node-results` 可另存逐节点 p 值，
该文件可能较大。历史结果保留在旧目录，不与新实验混用。

可用 `--repetitions`、`--null-graphs`、`--null-models`、`--alpha`、`--seed` 和 `--output` 修改运行参数：

```powershell
python -m experiment_support.experiment --null-graphs 100 --null-models gnm --alpha 0.01 --repetitions 10
```

先用小网络做流程检查时，可加 `--pilot`。该模式使用 200 个节点、两档边概率的正负例，
但仍可单独指定重复数和零图数。例如：

```powershell
python -m experiment_support.experiment --pilot --repetitions 3 --null-graphs 100 --output experiment_results/simulated_centrality_comparison_pilot
```

## 固定参数的连边概率倍率实验

```powershell
python -m experiment_support.ratio_sweep
```

它固定 `N=100`、`M=350`、中心节点数 `k=3`，依次测试倍率
`1、1.5、2、2.5、3、3.5、4、4.5、5、6`，每个倍率使用 50 个随机种子。
每张目标图**恰好有 350 条边**；每张目标图独立生成 1000 张 GNM 图形成零分布。
同一随机种子下的不同倍率使用相同的中心节点和随机边顺序，方便比较。

主要结果写入独立的 `experiment_results/ratio_sweep_per_target/` 目录：

- `summary.md`：各倍率的召回率、精确率、F1、平均报告数和平均误报数。
- `ratio_sweep.png`、`ratio_sweep.svg`：召回率、精确率和 F1 随倍率变化的趋势图。
- `replicates.csv`：500 张图的逐次结果。

可用 `--metric`、`--n`、`--m`、`--hubs`、`--ratios`、`--repetitions`、
`--null-graphs`、`--alpha`、`--seed` 和 `--output` 修改参数。例如：

```powershell
python -m experiment_support.ratio_sweep --n 100 --m 350 --hubs 3 --ratios 1 2 3 4 5 6
```

非度指标默认写入 `experiment_results/ratio_sweep_<metric>_per_target/`。绘图使用
matplotlib；若未安装，数值表格仍会生成。倍率实验的抽样方式和指标定义
见 [ratio_sweep.md](ratio_sweep.md)。

## 同图比较六种中心性指标

```powershell
python -m experiment_support.compare_metrics
```

此命令在相同随机种子的目标图上，为每张目标图分别生成 GNM 零模型图，比较六项中心性指标，
仅输出 Bonferroni 校正结果，比较召回率、精确率和 F1。运行后结果见 `experiment_results/metric_comparison_per_target/comparison.md`
与 `comparison.png`；各指标的详细结果位于同目录的指标子目录中。
可用 `--metrics degree lac` 只比较选定的指标。

如需安装非度指标与绘图依赖：

```powershell
python -m pip install networkx matplotlib
```

## 选择零模型

`experiment_support/ratio_sweep.py` 和 `experiment_support/compare_metrics.py` 接受
`--null-model gnm|configuration|chung_lu`，默认 `gnm`；模拟图方法比较入口用
`--null-models` 选择中心性实验的零模型，默认只用 GNM。例如：

```powershell
python -m experiment_support.ratio_sweep --metric lac --null-model configuration --repetitions 5 --null-graphs 1000
python -m experiment_support.compare_metrics --null-model chung_lu --metrics eigenvector gpg --repetitions 5 --null-graphs 1000
python -m experiment_support.experiment --null-models configuration chung_lu --repetitions 1 --null-graphs 1000
```

`gnm` 固定节点数和边数；当前倍率实验对每张目标图分别生成零分布，同一张目标图的六种指标使用相同的零图随机种子。
`configuration` 从每张目标图出发，随机执行保持简单图的双边交换；
每张零模型图都严格保持**每个对应节点**的度数。交换次数为每张图 `10 × M`
次尝试；这是配置模型的边交换近似抽样，不保证在具有该度序列的全部简单图上精确均匀。
因此使用度中心性时，配置模型下零模型图的度数集合与目标图相同；
在当前 Bonferroni 门槛下不会报告度异常节点。
`chung_lu` 根据目标图度数独立抽取每条边，使用
`min(d_u d_v / sum(d), 1)` 作为边概率；生成图为简单图，边数和度数可以变化，
输入度数在简单图中仅是近似期望值。

三种模型都分别对每张目标图建零分布，每张零模型图的**所有节点分数**都参与汇总。
100 个节点、1000 张零模型图得到 10 万个参考分数；经验 p 值使用
`(不低于观测分数的参考分数数目 + 1) / (参考分数总数 + 1)`。
这些分数来自 1000 张图，同一张图内的节点并不独立。
配置模型和 Chung–Lu 的汇总 p 值衡量的是相对随机节点整体的极端程度，
不能直接解释为固定某个节点度数后的逐节点显著性。
逐张目标图生成零分布会增加运行时间。
非 GNM 结果默认写入带模型名的新目录，避免覆盖已有结果。

GNM、配置模型与 Chung–Lu 六指标中心节点强度实验使用：

```powershell
python compare_null_models.py --workers 8
```

这条命令对每张目标图分别生成 GNM、配置模型、Chung–Lu 零分布，六种指标共享对应模型的零图。合并表格和图位于 `experiment_results/strength_n100_m350_k6_six_metrics_v1/comparison.md` 与 `comparison.png`。更改实验参数时应使用新的输出目录，避免混用检查点。
旧的 `null_model_comparison_pilot_10x100/` 和 `metric_comparison/` 结果保留作历史记录，其中 GNM 曾由所有目标图共用一套零分布；它们不参与新表格汇总。

默认采用 N=100、M=350、6 个植入节点、倍率 1、1.5、2、3、4、6、8、10；每倍率 10 张目标图，
每张目标图每种零模型各生成 100 张图，采用 DC、LAC、EC、BC、CC、GPG 六种指标。
上述命令将结果保存在 `experiment_results/strength_n100_m350_k6_six_metrics_v1/`，包括三个模型各自的
`comparison.md`、`comparison.csv`、`comparison.png`、`comparison.svg`、
`replicates.csv` 和 `node_results.csv`，以及三个模型的总比较表和图。总图按零模型分三行，
每行从左到右分别是 Recall、Precision、F1 Score，六条曲线分别对应六项指标。
每个指标先逐图计算，再对同一强度的 10 张图取平均；无报告时 Precision 和 F1 记为 0。
程序只计算 Bonferroni 校正 p 值，不计算原始 p 值。
每张目标图完成后会写入 `checkpoints/`；同参数重运行可从已有进度继续。
实验入口对目标图和零模型图使用相同的六指标实现。
先安装 `python -m pip install igraph`；可用 `--workers` 调整并行进程数。

## 基础正负例：20 张负例与 20 张正例

```powershell
python compare_basic_model_budgets.py --budgets 100 500 1000 --workers 8 --output experiment_results/basic_40_unique_graphs_aggregate
```

使用 20 组不同的 GNM 负例参数和 20 组不同的植入中心节点正例参数，
每组只生成一张图，总计 20 张负例图、20 张正例图。
参数列在入口文件的 `NEGATIVE_SPECS`、`POSITIVE_SPECS` 及结果目录的 `settings.json` 中。
同一张目标图使用 GNM、configuration、Chung–Lu 三种零模型；
DC、LAC、EC、BC、CC、GPG 共享各模型的零图。每个“目标图 × 零模型”生成一条 1000 张零图的
序列，前 100 张和前 500 张同时用于较小预算的结果。每张零图的全部节点分数
参与对应指标的经验零分布。

实验只计算 Bonferroni 校正 p 值。`summary.md` 和 `summary.csv` 是同一张汇总表，
按零模型、指标、零图数各列一行，报告负例误检率及正例 Recall、Precision、F1。
负例误检率先按每张图计算报告节点数除以节点数，再对 20 张负例图取平均；
正例三项指标先逐图计算，
再对 20 张正例图取平均。无报告时 Precision 和 F1 记为 0。
`summary.md` 下方另列 40 张图的节点预测合并后计算的整体 Recall、Precision、F1；
负例中的报告节点会计入整体 FP。文档末尾再单列零图数为 1000 的整体结果。
其后列出同预算下负例平均误检率和正例逐图平均 Recall、Precision、F1，方便与整体结果对照。
文档最后将零图数为 1000 时的上述指标合并为一张表。
每个目标图/零模型完成后写入检查点；同参数重运行可以续跑。
旧结果仍保存在 `experiment_results/basic_model_budget_comparison/` 和
`experiment_results/basic_reduced_model_budget_comparison/`。

## 中心结构：强度扫描与中心节点数量扫描

```powershell
python compare_central_structure.py --scenarios 60:90 100:350 150:900 --baseline-hubs 3 --ratios 1 1.5 2 2.5 3 3.5 4 4.5 5 6 --hub-counts 1 2 3 5 8 --count-ratio 4 --repetitions 50 --null-graphs 1000 --workers 8 --output experiment_results/central_structure_sweeps
```

三组 `(N,M)` 各进行两条扫描：强度扫描固定中心节点数 K=3、改变中心边与普通边概率比；
数量扫描固定概率比 4、改变 K。每个参数点使用 50 张配对目标图，
每张目标图分别生成 1000 张 GNM、配置模型、Chung–Lu 零图，六种中心性共享各模型零图。
只计算 Bonferroni 校正结果；单图无报告时 Precision、F1 记为 0。
同一个场景下两条扫描共有的 `K=3、概率比=4` 目标图只计算一次。

`report.md` 链接每个场景的强度和数量趋势图；每张图按零模型分三行，
每行从左到右是 Recall、Precision、F1 Score，六条曲线对应 DC、LAC、EC、BC、CC、GPG。
`summary.csv` 包含逐参数点均值、近似 95% 区间、平均检出/误报数、无报告占比，
以及对应的中心边和普通边概率；`replicates.csv` 保存逐图结果。
固定 N、M、概率比时增加 K 会改变两类边的实际概率，因此数量扫描反映
固定这些约束后的整体效果。完整实验约生成 630 万张零图，运行时间可能较长；
检查点支持同参数续跑。旧结果保存在原目录，不参与此次汇总。

只测试 `N=100、M=350`、固定概率比 2、增加中心节点数时，使用：

```powershell
python compare_central_structure.py --sweeps count --scenarios 100:350 --ratios 2 --hub-counts 1 2 3 5 8 --count-ratio 2 --repetitions 50 --null-graphs 1000 --workers 8 --output experiment_results/hub_count_ratio2_n100_m350
```

这条命令只生成数量扫描；固定的是中心相关边与普通边的概率比，
在 N、M 固定时两类边的实际概率会随中心节点数调整。

## 真实网络实验

`experiment_support/real_network_experiment.py` 读取 `network/*.txt` 边表，取每行前两列为节点，
忽略权重等附加列，把边转成无向、无权简单图，合并反向/重复边并删除自环。
节点数按文件中出现的端点计算；只有边表而没有节点清单时，无法恢复未出现的孤立节点。
零模型为 GNM，各目标图单独抽样；默认使用 DC、LAC、EC、BC、CC、GPG 六种指标，
每张目标图固定生成 100 张零图，不使用提前停止。

```powershell
python -m experiment_support.real_network_experiment --workers 4
```

该命令对 `network/*.txt` 中的 55 张真实网络使用六项指标和每网 100 张 GNM 零图，
结果写入 `experiment_results/real_network_six_metrics_gnm_b100_v1/`。
输出包含 `report.md`、`summary.csv`、`significant_nodes.csv`、`cleaning.csv`
和逐图检查点。文件没有真实中心节点标签，因此这些结果报告显著节点和运行时间，
不能直接计算分类任务的 Recall、Precision 或 F1。
此前带提前停止的结果保存在不带 `_full` 后缀的目录；请用新目录运行固定预算版本，
避免混用两种流程的检查点。

## 真实网络：中心性、零图数量与零模型对比

```powershell
python real_network_model_budget_comparison.py --metrics degree lac eigenvector betweenness closeness gpg --null-models gnm configuration chung_lu --budgets 100 1000
```

默认读取 `network/*.txt` 的全部真实网络。每张目标图分别拟合 GNM、严格保持各节点
度数的配置模型、Chung–Lu；每种模型生成 1000 张零图，前 100 张同时用于
100 张预算的结果。六种中心性共享同一张零图，计算校正后的显著节点集合。
不使用提前停止。输出目录为
`experiment_results/real_network_model_budget_comparison/`，包含：

- `summary.csv`：各网络、零模型、预算和中心性指标的报告数、报告比例、节点数、边数及耗时；
- `significant_nodes.csv`：报告节点及其经验 p 值和 Bonferroni 校正 p 值；
- `budget_comparison.csv`：100 张与 1000 张的报告集合差异；
- `model_comparison.csv`：同预算下两种零模型的报告集合交并和 Jaccard 相似度；
- `report.md`：按组合汇总的平均报告比例、报告集合稳定性及模型间一致性。

输入只有边表，没有真实中心节点标签，因此不计算 Recall、Precision、F1。
配置模型使用保持简单图且逐节点保度的边交换近似抽样；每张零图会再次检查
度序列。经验零分布汇总零图中**所有节点**的中心性分数，这反映相对于随机
节点整体的极端程度；配置模型和 Chung–Lu 的结果不能解释为固定某个节点
条件下的显著性。对每张零图都计算介数中心性，大图上的完整实验可能耗时很长。
运行过程中每完成一个“网络 × 零模型”会保存检查点，同参数重运行可续跑；
更改实验参数时请另设 `--output`。可以用 `--names` 选择网络，或用
`--max-nodes 200` 限制网络规模。


历史 HTC、SigEP 对比结果仅作存档；当前实验代码不再运行这两种方法。
