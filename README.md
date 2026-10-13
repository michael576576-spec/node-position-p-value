# 节点中心性模拟实验

在本项目目录下运行以下命令。需要 Python 3.10 或更新版本；两套实验使用
同一个经验 p 值算法，但目标图的生成方式不同。中心性指标由 `--metric`
指定，可选 `degree`、`harmonic`、`betweenness`、`pagerank`，默认是 `degree`。
非度指标需要 NetworkX，绘图需要 matplotlib。

`degree` 使用原始度数；在固定节点数的图上，将其除以 `N-1` 得到的标准化
度中心性只改变尺度，不改变排序或经验 p 值。其余三个指标分别使用 NetworkX
的调和中心性（不可达节点贡献为零）、标准化介数中心性和无向图 PageRank
（`alpha=0.85`）。指标计算集中在 [centrality.py](centrality.py)，GNM
零分布与 p 值检验集中在 [network_core.py](network_core.py)。新增指标时，
在 `centrality.py` 注册一个返回所有节点分数的函数即可。

## 原来的正例与负例实验

```powershell
python experiment.py
```

它运行 10 组 GNM 负例和 10 组植入中心节点的正例，每组使用 50 个随机种子。
每张目标图分别用 100 张匹配实际节点数和边数的 GNM 图计算经验 p 值。
原正例生成器固定的是**期望边数**，实际边数会波动。

主要结果：

- `experiment_results/basic_experiment/network_summary.md`：原始 p 值判定的 20 组汇总。
- `experiment_results/basic_experiment/bonferroni_comparison.md`：校正前后的报告数、检出数、误报数。
- `experiment_results/basic_experiment/negative_replicates.csv` 与
  `experiment_results/basic_experiment/positive_replicates.csv`：各组逐次结果。

可用 `--metric`、`--repetitions`、`--null-graphs`、`--alpha`、`--seed`
和 `--output` 修改运行参数。例如：

```powershell
python experiment.py --metric harmonic
```

非度指标默认写入 `experiment_results/<metric>_experiment/`，避免覆盖
度中心性的原实验结果。原实验为每张目标图重新生成零模型；介数中心性在完整
原实验上会花较长时间。要在相同图上快速比较多种指标，使用下方的
`compare_metrics.py`。

## 固定参数的连边概率倍率实验

```powershell
python ratio_sweep.py
```

它固定 `N=100`、`M=350`、中心节点数 `k=3`，依次测试倍率
`1、1.5、2、2.5、3、3.5、4、4.5、5、6`，每个倍率使用 50 个随机种子。
每张目标图**恰好有 350 条边**；所有目标图共用 1000 张 GNM 图形成的零分布。
同一随机种子下的不同倍率使用相同的中心节点和随机边顺序，方便比较。

主要结果写入独立的 `experiment_results/ratio_sweep/` 目录：

- `summary.md`：各倍率的召回率、精确率、F1、平均报告数和平均误报数。
- `ratio_sweep.png`、`ratio_sweep.svg`：召回率、精确率和 F1 随倍率变化的趋势图。
- `replicates.csv`：500 张图的逐次结果。

可用 `--metric`、`--n`、`--m`、`--hubs`、`--ratios`、`--repetitions`、
`--null-graphs`、`--alpha`、`--seed` 和 `--output` 修改参数。例如：

```powershell
python ratio_sweep.py --n 100 --m 350 --hubs 3 --ratios 1 2 3 4 5 6
```

非度指标默认写入 `experiment_results/ratio_sweep_<metric>/`。绘图使用
matplotlib；若未安装，数值表格仍会生成。倍率实验的抽样方式和指标定义
见 [ratio_sweep.md](ratio_sweep.md)。

## 同图比较四种中心性指标

```powershell
python compare_metrics.py
```

此命令在相同随机种子的目标图和 GNM 零模型图上，比较度、调和、介数、
PageRank 四种中心性，仅输出 Bonferroni 校正结果，比较召回率、精确率和 F1。运行后结果见 `experiment_results/metric_comparison/comparison.md`
与 `comparison.png`；各指标的详细结果位于同目录的指标子目录中。
可用 `--metrics degree harmonic` 只比较选定的指标。

如需安装非度指标与绘图依赖：

```powershell
python -m pip install networkx matplotlib
```

## 选择零模型

`experiment.py`、`ratio_sweep.py` 和 `compare_metrics.py` 都接受
`--null-model gnm|configuration|chung_lu`，默认 `gnm`。例如：

```powershell
python ratio_sweep.py --metric harmonic --null-model configuration --repetitions 5 --null-graphs 1000
python compare_metrics.py --null-model chung_lu --metrics harmonic pagerank --repetitions 5 --null-graphs 1000
python experiment.py --null-model configuration --metric betweenness --repetitions 1 --null-graphs 1000
```

`gnm` 固定节点数和边数，继续对同一组 N、M 的目标图共用零分布。
`configuration` 从每张目标图出发，随机执行保持简单图的双边交换；
每张零模型图都严格保持**每个对应节点**的度数。交换次数为每张图 `10 × M`
次尝试；这是配置模型的边交换近似抽样，不保证在具有该度序列的全部简单图上精确均匀。
因此使用度中心性时，配置模型下零模型图的度数集合与目标图相同；
在当前 Bonferroni 门槛下不会报告度异常节点。
`chung_lu` 根据目标图度数独立抽取每条边，使用
`min(d_u d_v / sum(d), 1)` 作为边概率；生成图为简单图，边数和度数可以变化，
输入度数在简单图中仅是近似期望值。

后两种模型分别对每张目标图建零分布，每张零模型图的**所有节点分数**都参与汇总。
100 个节点、1000 张零模型图得到 10 万个参考分数；经验 p 值使用
`(不低于观测分数的参考分数数目 + 1) / (参考分数总数 + 1)`。
这些分数来自 1000 张图，同一张图内的节点并不独立。
配置模型和 Chung–Lu 的汇总 p 值衡量的是相对随机节点整体的极端程度，
不能直接解释为固定某个节点度数后的逐节点显著性。
这两种模型须为每张目标图单独生成零分布，运行时间可能明显增加。
非 GNM 结果默认写入带模型名的新目录，避免覆盖已有结果。

`ratio_sweep_fast.py` 的精确超几何方法仅针对 GNM 的度分布，运行方式保持原样。

完整的配置模型与 Chung–Lu 四指标强度实验使用：

```powershell
python compare_null_models.py
```

先查看小规模趋势可运行：

```powershell
python compare_metrics.py --repetitions 10 --null-graphs 100 --output experiment_results/null_model_comparison_pilot_10x100/gnm
python compare_null_models.py --repetitions 10 --null-graphs 100 --workers 8 --output experiment_results/null_model_comparison_pilot_10x100
```

这两条命令在相同的 10 个目标图随机种子下，分别生成 GNM 参照结果以及配置模型、Chung–Lu 模型结果。合并表格和图位于 `experiment_results/null_model_comparison_pilot_10x100/comparison.md` 与 `comparison.png`。更改实验参数时应使用新的输出目录，避免混用检查点。

默认采用 N=100、M=350、3 个植入节点、10 个倍率、每倍率 50 张目标图、
每张目标图 1000 张零模型图，以及 DC、HC、BC、PR 四种指标。
结果保存在 `experiment_results/null_model_comparison/`，包括两个模型各自的
`comparison.md`、`comparison.csv`、`comparison.png`、`comparison.svg`、
`replicates.csv` 和 `node_results.csv`，以及包含已有 GNM 基线的总比较表和图。
每张目标图完成后会写入 `checkpoints/`；同参数重运行可从已有进度继续。
实验入口使用 igraph 的原生实现计算四种中心性，并用同一实现计算目标图和零模型图。
先安装 `python -m pip install igraph`；可用 `--workers` 调整并行进程数。
