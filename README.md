# Task1 — 二分类实习任务

> 本仓库在平台提供的 kNN 基线之上，系统性地对比了多个模型，并给出可复现的工程结构与完整文档。

---

## 一、任务说明（平台原文）

### 任务目标
用 Python 读取训练与测试 CSV，检查重复 ID，建立多数类与简单 kNN 基线，生成符合格式的提交。解释训练、验证和隐藏测试的区别。

### 评估与验收
Accuracy：预测标签完全正确的比例，越大越好。公开榜只使用部分隐藏测试数据；最终选择的提交在题目关闭并发布私榜后按剩余测试数据排名。不要利用公开榜反复推测标签。

### 数据说明
`train.csv` 有 `target`，`test.csv` 没有 `target`。评分用的真实标签仅保存在服务器。`sample_submission.csv` 是训练集均值/多数类参考提交，页面参考分对应此文件。`baseline.py` / `baseline.ipynb` 是可进一步对比的 kNN 起点，在本地运行并生成 `submission.csv`。缺失值、标准化和调参只使用训练集。

---

## 二、项目简介

- **任务类型**：二分类（`target ∈ {0, 1}`）
- **数据规模**：训练 156 行，测试 84 行，共 4 个数值特征 `f1~f4`
- **评估协议**：全部实验统一使用 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`，报告 CV 均值与标准差
- **防泄漏**：所有标准化、多项式扩展、缺失值填补都包装在 `sklearn.Pipeline` 内，严格在每一折的训练部分上拟合
- **选榜策略**：**永远以 CV 最优为准**，不根据公开榜反复试探

本仓库在原始 kNN 基线（CV accuracy = 0.8268）之上，依次完成了 Logistic Regression、SVM、GBDT 以及三者的融合实验，最终 CV accuracy 达到 **0.9363**。

---

## 三、项目结构

```
Task1/
├── README.md                       # 本文档
├── requirements.txt                # Python 依赖
├── .gitignore
│
├── data/                           # 原始数据（只读）
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── baseline.py                     # 平台原始 kNN 基线（已修补路径）
├── baseline.ipynb                  # 对应的 Notebook
│
├── src/                            # 项目公共代码
│   ├── __init__.py
│   ├── data.py                     # 统一的数据加载 / 特征拆分
│   ├── evaluate.py                 # 5 折 CV 工具、submission 写入
│   └── models/                     # 预留给可复用的自定义模型封装
│
├── experiments/                    # 每个模型一个独立可复现脚本
│   ├── exp01_logreg.py             # Logistic Regression
│   ├── exp01_logreg_grid.csv       #   -> 完整网格结果
│   ├── exp02_svm.py                # Support Vector Classifier
│   ├── exp02_svm_grid.csv
│   ├── exp03_gbdt.py               # Gradient Boosted Decision Trees
│   ├── exp03_gbdt_grid.csv
│   ├── exp04_ensemble.py           # Voting / Stacking 融合
│   └── exp04_ensemble_grid.csv
│
├── submissions/                    # 每次实验独立保存一份提交
│   ├── submission_logreg.csv
│   ├── submission_svm.csv
│   ├── submission_gbdt.csv
│   └── submission_ensemble.csv
│
├── submission.csv                  # 当前最佳提交（镜像自 submissions/ 中 CV 最优者）
│
└── docs/                           # 每个阶段的中文说明文档
    ├── 01_eda.md                   # 数据探索
    ├── 02_logreg.md                # Logistic Regression 原理 + 调参
    ├── 03_svm.md                   # SVM 原理 + 调参
    ├── 04_gbdt.md                  # GBDT 原理 + 调参
    └── 05_ensemble.md              # 融合方案与最终结论
```

### 关于 `submission.csv`
- `submissions/submission_<model>.csv`：**每个实验独立输出**，便于回溯和事后融合，不会互相覆盖
- `submission.csv`（根目录）：**当前 CV 最优的那一份的镜像**，用来一键提交
- 只有当新模型的 CV 均值 **严格高于** 当前最优时，才会调用 `make_submission(..., mirror_as_best=True)` 覆盖根目录文件

---

## 四、环境与运行

### 1 安装依赖（如果重装环境）

```powershell
python -m pip install -r requirements.txt
```

### 2 运行各个实验

从仓库根目录执行：

```powershell
# 平台基线（kNN，作为对照）
python baseline.py

# 四个主实验（任意顺序，彼此独立）
python experiments\exp01_logreg.py
python experiments\exp02_svm.py
python experiments\exp03_gbdt.py
python experiments\exp04_ensemble.py
```

每个脚本会在屏幕上打印完整网格的 Top-10、最终选中的超参，以及 CV 均值 ± 标准差；同时写入
`experiments/exp0X_*_grid.csv` 保存完整网格，`submissions/submission_<model>.csv` 保存该模型的预测。

---

## 五、模型效果汇总

> 评估指标：`StratifiedKFold(5, shuffle=True, random_state=42)` 下的 accuracy。

| 排名 | 模型 | CV 均值 | CV 标准差 | 相对基线 | 当前提交 |
|:--:|---|---:|---:|---:|:--:|
| 🥇 | **Hard-vote (LogReg + SVM + GBDT)** | **0.9363** | 0.0445 | **+10.95** | ✅ `submission.csv` |
| 🥈 | LogReg (poly=2, L2, C=1) | 0.9296 | **0.0235** | +10.28 | `submissions/submission_logreg.csv` |
| 🥉 | SVM (linear, C=10) | 0.9234 | 0.0516 | +9.66  | `submissions/submission_svm.csv` |
| 4 | GBDT (sklearn GB, 400 棵 × depth 4) | 0.8982 | 0.0694 | +7.14 | `submissions/submission_gbdt.csv` |
| 5 | 基线 kNN(k=5) | 0.8268 | 0.0635 | — | — |

### 关键结论

1. **数据边界近似线性**：`f1` 与 target 相关系数 0.61、`f2` 为 0.45，`f3`/`f4` 近似噪声。LogReg + 二次多项式即可拟合得很好；SVM 的最佳核也退化为线性核，GBDT 没能提供额外收益。
2. **样本极少（156 行）**：复杂模型（深树、更大森林、更高次多项式）都会显著增大方差。LogReg 的 `std=0.0235` 是全场最低。
3. **融合提升主要来自偏差互补**：Hard-vote 比单一 LogReg 高 0.7 个点，但方差翻倍。只有在接受「略高方差换略高均值」时才值得提交。如需稳健提交，可将 `submission.csv` 替换为 `submissions/submission_logreg.csv`。
4. **Soft-vote 弱于 Hard-vote**：GBDT 输出的概率校准较差，软投票反而被它的高置信错误拖累；Stacking 在 156 行上元学习器也容易欠拟合。

详细推导、网格范围与分析见 `docs/` 下的 5 篇中文文档。

---
