---
title: "吴恩达机器学习课程(4)-Linear Regression with Multiple Variables"
date: "2020-02-26T16:22:43+08:00"
updated: "2021-10-08T23:19:13+08:00"
categories: ["视频学习", "机器学习"]
tags: ["机器学习", "线性回归"]
author: "DPer"
original_url: "http://forestneo.top/2020/02/26/ML-吴恩达机器学习课程-04-Linear Regression with Multiple Variables/"
source_html: "ForestNeo-website-master/2020/02/26/ML-吴恩达机器学习课程-04-Linear Regression with Multiple Variables/index.html"
mathjax: true
---

本章节讲逻辑回归，组织结构如下：

- Linear Regression with Multiple Variables

  - Multiple Features

  - Gradient Descent for Multiple Variables

  - Gradient Descent in Practice I - Feature Scaling

  - Gradient Descent in Practice II - Learning Rate

  - Features and Polynomial Regression

  - Normal Equation

  - Normal Equation Noninvertibility (Optional)

<a id="Multiple-Features"></a>

# Multiple Features

目前为止，我们探讨了单变量/特征的回归模型，现在我们对房价模型增加更多的特征，例如房间数楼层等，构成一个含有多个变量的模型，模型中的特征为 $\left( {x_{1}},{x_{2}},…,{x_{n}} \right)$。

![](<https://forest-pic.oss-cn-beijing.aliyuncs.com/20200430093130.png>)

增添更多特征后，我们引入一系列新的注释：

- $n$ 代表特征的数量

- ${x^{\left( i \right)}}$代表第 $i$ 个训练实例，是特征矩阵中的第$i$行，是一个**向量**（**vector**）。

- ${x}_{j}^{\left( i \right)}$代表特征矩阵中第 $i$ 行的第 $j$ 个特征，也就是第 $i$ 个训练实例的第 $j$ 个特征。如上图的$x_{2}^{\left( 2 \right)}=3,x_{3}^{\left( 2 \right)}=2$。

- 假设 $h$ 表示为：$h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+…+{\theta_{n}}{x_{n}}$，这个公式中有$n+1$个参数和$n$个变量，为了使得公式能够简化一些，引入$x_{0}=1$，则公式转化为：$h_{\theta} \left( x \right)={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+…+{\theta_{n}}{x_{n}}$

此时模型中的参数是一个$n+1$维的向量，任何一个训练实例也都是$n+1$维的向量，特征矩阵$X$的维度是 $m\times(n+1)$。 因此公式可以简化为：$h_{\theta} \left( x \right)={\theta^{T}}X$，其中上标$T$代表矩阵转置。

<a id="Gradient-Descent-for-Multiple-Variables"></a>

# Gradient Descent for Multiple Variables

与单变量线性回归类似，在多变量线性回归中，我们也构建一个代价函数，则这个代价函数是所有建模误差的平方和，即假设$h_{\theta}\left( x \right)=\theta^{T}X={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+…+{\theta_{n}}{x_{n}}$，那么：

$$
J\left( {\theta_{0}},{\theta_{1}},...,{\theta_{n}} \right) = \frac{1}{2m}\sum_{i=1}^m\left[h_\theta(x^{(i)})-y^{(i)}\right]^2
$$

我们的目标和单变量线性回归问题中一样，是要找出使得代价函数最小的一系列参数。  
多变量线性回归的批量梯度下降算法为：

$$
\theta_{j}=\theta_{j}-a \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}
$$

<a id="Gradient-Descent-in-Practice-I-Feature-Scaling"></a>

# Gradient Descent in Practice I - Feature Scaling

在我们面对多维特征问题的时候，我们要保证这些特征都具有相近的尺度，这将帮助梯度下降算法更快地收敛。

以房价问题为例，假设我们使用两个特征，房屋的尺寸和房间的数量，尺寸的值为 0-2000平方英尺，而房间数量的值则是0-5，以两个参数分别为横纵坐标，绘制代价函数的等高线图能，看出图像会显得很扁，梯度下降算法需要非常多次的迭代才能收敛。解决的方法是尝试将所有特征的尺度都尽量缩放到-1到1之间。如图：

![](<https://forest-pic.oss-cn-beijing.aliyuncs.com/20200430093235.png>)

最简单的方法是令：$x_n = \frac{x_n-\mu_n}{s_n}$，其中 $\mu_n$是平均值，$s_n$是标准差。

<a id="Gradient-Descent-in-Practice-II-Learning-Rate"></a>

# Gradient Descent in Practice II - Learning Rate

梯度下降算法收敛所需要的迭代次数根据模型的不同而不同，我们不能提前预知，我们可以绘制迭代次数和代价函数的图表来观测算法在何时趋于收敛。

![](<https://forest-pic.oss-cn-beijing.aliyuncs.com/20200430093350.png>)

也有一些自动测试是否收敛的方法，例如将代价函数的变化值与某个阀值（例如0.001）进行比较，但通常看上面这样的图表更好。

梯度下降算法的每次迭代受到学习率的影响，如果学习率$a$过小，则达到收敛所需的迭代次数会非常高；如果学习率$a$过大，每次迭代可能不会减小代价函数，可能会越过局部最小值导致无法收敛。通常可以考虑尝试些学习率：$\alpha=0.01，0.03，0.1，0.3，1，3，10$。

<a id="Features-and-Polynomial-Regression"></a>

# Features and Polynomial Regression

仍然以房价预测为例：

![](<https://forest-pic.oss-cn-beijing.aliyuncs.com/20200430093442.png>)

其假设可以为：$h_{\theta}(x)=\theta_{0}+\theta_{1}\times x_1+\theta_{2}\times x_2$。线性回归并不适用于所有数据，有时我们需要曲线来适应我们的数据，比如一个二次方模型：$h_{\theta}( x )=\theta_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}^2$  
 或者三次方模型： $h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}^2}+{\theta_{3}}{x_{3}^3}$。

![](<https://forest-pic.oss-cn-beijing.aliyuncs.com/20200430094130.png>)

通常我们需要先观察数据然后再决定准备尝试怎样的模型。 另外，我们可以令：${x}_{2}=x_{2}^{2},{x}_{3}=x_{3}^{3}$，从而将模型转化为线性回归模型。根据函数图形特性，我们还可以使：$h_{\theta}(x)=\theta_{0}+\theta_{1}(s i z e)+\theta_{2}(s i z e)^{2}$，或者：${h}_{\theta}(x)={\theta }_{0}\text{+}{\theta }_{1}(size)+{\theta }_{2}\sqrt{size}$。注：如果我们采用多项式回归模型，在运行梯度下降算法前，特征缩放非常有必要。

<a id="Normal-Equation"></a>

# Normal Equation

到目前为止，我们都在使用梯度下降算法，但是对于某些线性回归问题，正规方程方法是更好的解决方案。如：

![](<https://forest-pic.oss-cn-beijing.aliyuncs.com/20200430094316.png>)

正规方程是通过求解下面的方程来找出使得代价函数最小的参数的：$\frac{\partial}{\partial \theta_{j}} J\left(\theta_{j}\right)=0$ 。  
 假设我们的训练集特征矩阵为 $X$（包含了 ${x}_{0}=1$）并且我们的训练集结果为向量 $y$，则利用正规方程解出向量 $\theta ={\left( {X^T}X \right)}^{-1}{X^{T}}y$ 。

<a id="Normal-Equation-Noninvertibility-Optional"></a>

# Normal Equation Noninvertibility (Optional)

略
