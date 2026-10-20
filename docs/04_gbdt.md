# 04 - Gradient Boosted Decision Trees（梯度提升决策树）

## 为什么尝试 GBDT

尽管 EDA 显示边界接近线性，GBDT 仍是表格数据上的事实标准基线：非参数、
自动捕获特征交互、对量纲不敏感，通常可以给出"这类任务的强下界"。我们跑一次
是为了确认前两轮 LogReg / SVM 的结果**没有漏掉某种非线性结构**。

同时并行对比两种实现：

* `sklearn.ensemble.GradientBoostingClassifier` —— 教科书式的纯 Python 实现；
* `lightgbm.LGBMClassifier` —— 工业界常用的基于直方图的快速提升树。

## 网格搜索

评估协议与其他实验保持一致：
`StratifiedKFold(5, shuffle=True, random_state=42)`。

| 超参数 | 取值 | 设置理由 |
|---|---|---|
| `n_estimators` | 100 – 800 | 轮数越多容量越大，但 156 行上的过拟合风险也上升。 |
| `learning_rate` | 0.03, 0.05, 0.1 | 一般"小 lr + 多轮"更稳。 |
| `max_depth`（sklearn） / `num_leaves`（lgbm） | depth 2–4 / leaves 7, 15, 31 | 小数据必须用浅树。 |
| `min_child_samples`（lgbm） | 5, 10, 20 | 保证每个叶子有足够样本，是强正则。 |
| `subsample`, `colsample_bytree` | 0.9 | 温和的随机正则。 |
| `reg_lambda`（lgbm） | 1.0 | 叶权重的 L2 正则。 |

Top 结果（完整表格：`experiments/exp03_gbdt_grid.csv`）：

| 实现 | n_est | max_depth | lr | num_leaves | min_child | CV 均值 | CV 标准差 |
|:---:|---:|:---:|---:|:---:|:---:|---:|---:|
| **sk-gb** | **400** | **4** | **0.03** | – | – | **0.8982** | 0.0694 |
| sk-gb | 200 | 2 | 0.05 | – | – | 0.8980 | 0.0642 |
| sk-gb | 400 | 3 | 0.03 | – | – | 0.8980 | 0.0642 |
| sk-gb | 100 | 2 | 0.10 | – | – | 0.8915 | 0.0582 |
| lgbm  | 200 | – | 0.10 | 7 | 10 | 0.8786 | 0.0540 |

## 结论对比

| 模型 | 5 折 CV accuracy | CV 标准差 |
|---|---:|---:|
| 基线 kNN(5) | 0.8268 | 0.0635 |
| LogReg (poly=2, L2, C=1) | **0.9296** | **0.0235** |
| SVM（线性/RBF 并列最佳） | 0.9234 | 0.0516 |
| GBDT（sklearn GB 最优） | 0.8982 | 0.0694 |

* **GBDT 落后两种线性模型约 3 个点**，方差却约为 LogReg 的 3 倍。156 行数据下，
  每棵树只能看到约 125 个样本，即使深度为 2 的桩，上百棵堆叠后也可能把噪声
  当成信号。
* **LightGBM 弱于 sklearn-GB**：LightGBM 的 leaf-wise 生长更激进，需要更大
  数据量才能发挥。
* **进一步证实 EDA 的直觉**：主要信号是 `f1`/`f2` 的近似线性贡献，额外的树
  灵活度只在训练噪声上发挥作用。因此 GBDT 的预测保留下来**仅作为融合输入**
  （见 `05_ensemble.md`），不会被提升为 `submission.csv`。

## 产出文件

* 脚本：`experiments/exp03_gbdt.py`
* 完整网格：`experiments/exp03_gbdt_grid.csv`
* 预测提交：`submissions/submission_gbdt.csv`（未被提升为当前最佳）。
