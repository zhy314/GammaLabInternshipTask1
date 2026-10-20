# 02 - Logistic Regression（逻辑回归）

## 为什么优先尝试逻辑回归

根据 `01_eda.md` 的结论：`f1`（ρ ≈ 0.61）和 `f2`（ρ ≈ 0.45）表现出很强的
**线性信号**，数据集规模很小（156 行），特征本身也接近标准化。在这种设定下，
一个带正则的线性模型通常非常难被击败：

* **线性边界方差低**：每一折验证只有约 31 个样本，模型稳定性比表达力更重要；
* **L2 正则项**提供一个清晰的偏差/方差旋钮（`C = 1/λ`）；
* 加一层较低阶的 **多项式扩展（degree=2）** 可以判断：是否存在轻度非线性
  边界，而不必立刻跳到完全非参数模型。

模型封装为 `Pipeline(StandardScaler, [PolynomialFeatures, StandardScaler], LogisticRegression)`，
以保证每一折**都只在训练部分**上拟合标准化参数，不会泄漏验证数据的信息。

## 网格搜索

评估协议：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`，指标为
accuracy。同一协议下重新测得的基线 `kNN(k=5)` 为 **0.8268 ± 0.0635**。

| 超参数 | 取值 | 设置理由 |
|---|---|---|
| `poly`（`PolynomialFeatures` 的阶数） | 1, 2, 3 | 1 = 纯线性；2 = 加入两两交互与平方项；3 = 考察更高灵活度是否带来提升还是过拟合。 |
| `penalty` | `l2`, `l1` | `l2` 是标准安全选择；`l1` 自带特征选择，可验证 `f3`/`f4` 是否纯噪声。 |
| `C`（正则强度的倒数） | 0.01, 0.1, 0.5, 1, 2, 5, 10, 100 | 覆盖从强正则到几乎无正则共 4 个数量级。 |

网格 Top 结果（完整表格见 `experiments/exp01_logreg_grid.csv`）：

| poly | penalty | C | CV 均值 | CV 标准差 |
|---:|:---:|---:|---:|---:|
| **2** | **l2** | **1.0** | **0.9296** | **0.0235** |
| 1 | l2 | 0.1 | 0.9169 | 0.0475 |
| 2 | l2 | 0.5 | 0.9169 | 0.0429 |
| 1 | l2 | 0.5 | 0.9167 | 0.0436 |
| 1 | l2 | 10 | 0.9167 | 0.0327 |

## 最终选用配置

```
StandardScaler
PolynomialFeatures(degree=2, include_bias=False)
StandardScaler
LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000)
```

* **5 折 CV accuracy**：**0.9296 ± 0.0235**
* **相对基线 kNN(5) 提升**：+10.28 个点，且方差显著更小。
* **观察**：
  * `degree=3` 没有继续提升，反而让方差变大——说明二次项已经捕捉到了真实边界。
  * `l1` 与 `l2` 在大多数 `C` 上基本持平，说明四个特征都至少有些微贡献，没有
    任何一个被稳定地置零。
  * 模型非常稳定（std ≈ 0.024），这在公开榜只采样部分测试集的情况下尤为重要。

## 产出文件

* 脚本：`experiments/exp01_logreg.py`
* 完整网格：`experiments/exp01_logreg_grid.csv`
* 预测提交：`submissions/submission_logreg.csv`（并镜像到仓库根 `submission.csv`
  作为当前最佳）。
