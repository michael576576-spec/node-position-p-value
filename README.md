# 节点中心性模拟实验

在本项目目录下运行以下命令。需要 Python 3.10 或更新版本；两套实验使用
同一个节点度经验 p 值算法，但目标图的生成方式不同。

## 原来的正例与负例实验

```powershell
python experiment.py
```

它运行 10 组 GNM 负例和 10 组植入中心节点的正例，每组使用 50 个随机种子。
每张目标图分别用 100 张匹配实际节点数和边数的 GNM 图计算经验 p 值。
原正例生成器固定的是**期望边数**，实际边数会波动。

主要结果：

- `experiment_results/network_summary.md`：原始 p 值判定的 20 组汇总。
- `experiment_results/bonferroni_comparison.md`：校正前后的报告数、检出数、误报数。
- `experiment_results/negative_replicates.csv` 与
  `experiment_results/positive_replicates.csv`：各组逐次结果。

可用 `--repetitions`、`--null-graphs`、`--alpha`、`--seed` 和 `--output`
修改运行参数。例如 `python experiment.py --seed 123`。

## 固定参数的连边概率倍率实验

```powershell
python ratio_sweep.py
```

它固定 `N=100`、`M=350`、中心节点数 `k=3`，依次测试倍率
`1、1.5、2、2.5、3、3.5、4、4.5、5、6`，每个倍率使用 50 个随机种子。
每张目标图**恰好有 350 条边**；所有目标图共用 1000 张 GNM 图形成的零分布。
同一随机种子下的不同倍率使用相同的中心节点和随机边顺序，方便比较。

主要结果写入独立的 `experiment_results/ratio_sweep/` 目录：

- `summary.md`：各倍率的检出率、平均报告数、平均误报数和完全恢复率。
- `ratio_sweep.png`、`ratio_sweep.svg`：指标随倍率变化的趋势图。
- `replicates.csv`：500 张图的逐次结果。

可用 `--n`、`--m`、`--hubs`、`--ratios`、`--repetitions`、
`--null-graphs`、`--alpha`、`--seed` 和 `--output` 修改参数。例如：

```powershell
python ratio_sweep.py --n 100 --m 350 --hubs 3 --ratios 1 2 3 4 5 6
```

绘图使用 matplotlib；若未安装，数值表格仍会生成。倍率实验的抽样方式和
指标定义见 [ratio_sweep.md](ratio_sweep.md)。
