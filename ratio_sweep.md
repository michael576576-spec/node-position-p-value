# 中心节点连边倍率实验

运行 `python ratio_sweep.py`。可用 `--metric degree|harmonic|betweenness|pagerank`
选择中心性指标，默认是 `degree`。默认固定节点数 `N=100`、边数 `M=350`、
中心节点数 `k=3`，考察倍率 `1、1.5、2、2.5、3、3.5、4、4.5、5、6`。
每个倍率生成 50 张图，随机种子默认是 42。命令行参数 `--n`、`--m`、
`--hubs`、`--ratios`、`--repetitions`、`--null-graphs`、`--alpha`、
`--seed` 和 `--output` 可调整实验设置。

## 生成模型

每张图恰好有 `M` 条边。设 `A` 为普通节点之间的候选边数量，`B` 为至少
接触一个中心节点的候选边数量。给定倍率 `r`，两类候选边的**边际**连边概率为

```text
p_ordinary = M / (A + r B)
p_hub      = r p_ordinary
```

脚本先抽取中心相关边数 `H`，再分别从两类候选边中均匀抽边。`H` 的分布经
数值校准，使平均概率之比等于 `r`。因此，所有图的 `N`、`M`、`k` 固定，
而 `r=1` 时的生成模型恰好是 GNM(N,M)。固定 `M` 会使图中的边彼此相关，
上式表示跨随机图的边际概率。

同一重复编号在各倍率下使用同一组中心节点及相同的随机边顺序，形成配对
比较。零分布由 1000 张 GNM(N,M) 图构成，在所有倍率和重复之间共用。
零分布和目标图均使用所选中心性指标。只输出 Bonferroni 校正 p 值，
判定条件为校正 p 值 `< 0.05`（等价于未校正 p 值 `< 0.05/N`）。

## 输出

度中心性的默认输出位于 `experiment_results/ratio_sweep/`；其他指标默认
输出到 `experiment_results/ratio_sweep_<metric>/`：

- `summary.md`、`summary.csv`：各倍率的召回率、精确率、F1、平均报告数和
  平均误报数，仅列出 Bonferroni 判定结果。
- `ratio_sweep.png`、`ratio_sweep.svg`：召回率、精确率和 F1
  的趋势图。绘图需要 matplotlib；若未安装，表格仍会生成。
- `replicates.csv`：500 张图各自的统计结果。
- `node_results.csv`、`network_edges.csv`：逐节点判定和每张图的边列表。

每张图分别计算召回率、精确率和 F1，再对同倍率的 50 张图取平均。
单图没有报告节点时，精确率和 F1 记为 0。
