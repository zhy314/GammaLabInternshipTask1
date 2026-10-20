# 任务目标
用 Python 读取训练与测试 CSV，检查重复 ID，建立多数类与简单 kNN 基线，生成符合格式的提交。解释训练、验证和隐藏测试的区别。

# 评估与验收
Accuracy：预测标签完全正确的比例，越大越好。公开榜只使用部分隐藏测试数据；最终选择的提交在题目关闭并发布私榜后按剩余测试数据排名。不要利用公开榜反复推测标签。

# 数据说明
train.csv 有 target，test.csv 没有 target。评分用的真实标签仅保存在服务器。sample_submission.csv 是训练集均值/多数类参考提交，页面参考分对应此文件。baseline.py / baseline.ipynb 是可进一步对比的kNN起点，在本地运行并生成 submission.csv。缺失值、标准化和调参只使用训练集。
