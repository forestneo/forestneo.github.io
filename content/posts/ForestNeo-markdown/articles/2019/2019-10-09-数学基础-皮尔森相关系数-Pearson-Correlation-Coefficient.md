---
title: "数学基础-皮尔森相关系数 (Pearson Correlation Coefficient)"
date: "2019-10-09T20:59:42+08:00"
updated: "2020-05-27T10:33:13+08:00"
categories: ["数学基础"]
tags: ["Probabilistic Tools"]
author: "DPer"
original_url: "http://forestneo.top/2019/10/09/数学基础-皮尔森相关系数-Pearson-Correlation-Coefficient/"
source_html: "ForestNeo-website-master/2019/10/09/数学基础-皮尔森相关系数-Pearson-Correlation-Coefficient/index.html"
mathjax: true
---

<a id="用途"></a>

# 用途

皮尔森相关系数用来反应两个变量`线性相关`的程度，用 $r$ 表示。相关性分为以下三种：

- $r=0$：两个变量不相关。

- $r>0$：Y 与 X 呈正相关，即 Y 随着 X 的增大有增大的趋势。越接近 1，正相关程度越大。

- $r<0$：Y 与 X 呈负相关，即 Y 随着 X 的增大有减小的趋势。越接近 -1，负相关程度越大。

比如，在现实生活中，数据会倾向于认为，人们的消费水平和收入水平呈正相关。这时候就可以勇皮尔森系数去计算这个相关性。

<a id="数学计算"></a>

# 数学计算

皮尔森相关系数一般用 $r$ 表示。该值越大，表明相关性越强。其定义为两个变量之间的协方差和标准差的商，即：

$$
\rho_{X, Y}=\frac{\operatorname{cov}(X, Y)}{\sigma_{X} \sigma_{Y}}=\frac{E\left[\left(X-\mu_{X}\right)\left(Y-\mu_{Y}\right)\right]}{\sigma_{X} \sigma_{Y}}
$$

上述定义的是总体的相关系数，对于样本来说，通过估算样本的协方差和标准差，可以得到羊背皮尔森系数，常用小写字母 $r$ 表示，为：

$$
r=\frac{\sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)\left(Y_{i}-\bar{Y}\right)}{\sqrt{\sum_{i=1}^{n}\left(X_{i}-\bar{X}\right)^{2}} \sqrt{\sum_{i=1}^{n}\left(Y_{i}-\bar{Y}\right)^{2}}}
$$

<a id="代码"></a>

# 代码

```python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/9
# @Author  : ForestNeo
# @Site    : forestneo.com
# @Email   : dr.forestneo@gmail.com
# @Software: PyCharm

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 产生数据并给数据加上一定噪声
    x, y = np.arange(100), np.arange(100)
    y_1 = y + np.random.uniform(low=-50, high=50, size=len(y))
    y_2 = y + np.random.uniform(low=-10, high=10, size=len(y))

    # 计算皮尔森相关系数
    pear_rate_1, pear_rate_2 = stats.pearsonr(x, y_1), stats.pearsonr(x, y_2)

    # 画图
    fig = plt.figure(figsize=(12, 5))
    ax_1, ax_2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

    ax_1.scatter(x, y_1)
    ax_2.scatter(x, y_2)
    title_1 = "pearson correlation = " + str(pear_rate_1[0])
    title_2 = "pearson correlation = " + str(pear_rate_1[0])
    ax_1.set_title(title_1)
    ax_2.set_title(title_2)
    plt.show()
```

实验结果如下：

![fig01](<https://forest-pic.oss-cn-beijing.aliyuncs.com/20200527103307.png>)
