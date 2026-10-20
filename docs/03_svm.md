# 03 - Support Vector Classifier（支持向量机）

## 为什么尝试 SVM

训练集仅有 156 行、4 个数值特征，这是教科书式的 SVM 适用场景：

* **最大间隔目标**自带正则效果，在小样本上泛化通常不错；
* **RBF 核**可以用隐式方式描绘任意光滑的非线性边界，与前一轮的
  LogReg + 多项式扩展形成互补；
* 顺带做一次 **线性核** 对照：如果 RBF 并不优于线性核，就说明问题本质确实接近
  线性；
* **多项式核**对比 LogReg 使用的 `PolynomialFeatures` 展开，可做一次直接的对照。

所有特征均在 Pipeline 内部完成标准化，以保证 RBF 核的距离计算在各维度上是
**各向同性**的。

## 网格搜索

评估协议与前一轮完全相同：`StratifiedKFold(5, shuffle=True, random_state=42)`。

| 超参数 | 取值 | 设置理由 |
|---|---|---|
| `kernel` | `linear`、`rbf`、`poly` | 覆盖线性、局部灵活与全局多项式三类边界。 |
| `C` | 0.01 ~ 100（对数均匀） | 在"间隔宽度"与"训练误差"之间权衡。 |
| `gamma`（RBF/poly） | `scale`、`auto`、0.05, 0.1, 0.25, 0.5, 1, 2 | 决定每个支持向量的"影响半径"：γ 小 → 光滑；γ 大 → 边界剧烈。 |
| `degree`（poly 专用） | 2, 3 | 与 LogReg 的多项式阶数呼应。 |

Top 结果（完整表格：`experiments/exp02_svm_grid.csv`）：

| kernel | C | gamma | degree | CV 均值 | CV 标准差 |
|:---:|---:|:---:|:---:|---:|---:|
| **linear** | **10** | – | – | **0.9234** | 0.0516 |
| rbf | 0.5 | auto | – | 0.9234 | 0.0516 |
| rbf | 0.5 | scale | – | 0.9234 | 0.0516 |
| rbf | 0.5 | 0.25 | – | 0.9234 | 0.0516 |
| linear | 100 | – | – | 0.9169 | 0.0556 |

RBF 与线性核得分完全持平（γ 较小时 RBF 近似退化为线性核），这进一步确认了
**决策边界非常接近线性**。我们选择 **线性核 + `C=10`** 作为代表性配置（精度相同
但模型更简单）。

## 结论对比

| 模型 | 5 折 CV accuracy | CV 标准差 |
|---|---:|---:|
| 基线 kNN(5) | 0.8268 | 0.0635 |
| LogReg (poly=2, L2, C=1) | **0.9296** | **0.0235** |
| SVM（线性/RBF 并列最佳） | 0.9234 | 0.0516 |

SVM 达到了与 LogReg 相近的水平，但方差大约是 LogReg 冠军的 2 倍，因此仓库根目录
的 `submission.csv` 仍然指向 LogReg 的预测；SVM 的预测写入
`submissions/submission_svm.csv` 以便后续融合（见 `05_ensemble.md`）。

## 产出文件

* 脚本：`experiments/exp02_svm.py`
* 完整网格：`experiments/exp02_svm_grid.csv`
* 预测提交：`submissions/submission_svm.csv`（未被提升为当前最佳）。
