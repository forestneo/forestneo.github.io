---
title: "论文阅读-The Prevention and Handling of the Missing Data"
date: "2019-11-11T18:42:46+08:00"
updated: "2021-12-09T10:40:22+08:00"
categories: ["论文阅读"]
tags: ["Missing Data"]
author: "DPer"
original_url: "http://forestneo.top/2019/11/11/PAPER-The Prevention and Handling of the Missing Data/"
source_html: "ForestNeo-website-master/2019/11/11/PAPER-The Prevention and Handling of the Missing Data/index.html"
---

文章标题：The Prevention and Handling of the Missing Data  
作者：Hyun Kang  
作者单位：Department of Anesthesiology and Pain Medicine, Chung-Ang Universtiy College of Medicine, Seoul, Korea  
发表年份：2013

---

<a id="Types-of-Missing-Data"></a>

# Types of Missing Data

根据丢失数据的原因，Robin将丢失数据分文以下三种类型

- MCAR: Missing Completely at Random

- MAR: Missing at Random

- MNAR: Missing not at Random

<a id="Missing-Completely-at-Random"></a>

## Missing Completely at Random

> Missing completely at random (MCAR) is defined as when  
> the probability that the data are missing is not related to either the specific value which is supposed to be obtained or the set of observed responses.

在 MCAR 的情况下，由于丢失数据是完全随机的，数据的统计特性不受影响。

<a id="MAR-Missing-at-Random"></a>

## MAR: Missing at Random

> Missing at random (MAR) is a more realistic assumption for  
> the studies performed in the anesthetic (麻醉学) field.

由于这篇文章一开始是发表在生物相关的期刊，所以涉及到麻醉学上面相关的概念。

<a id="MNAR-Missing-not-at-Random"></a>

## MNAR: Missing not at Random

> If the characters of the data do not meet those of MCAR or  
> MAR, then they fall into the category of missing not at random (MNAR).

除了上面的两类，剩下的都归为第三类了。

<a id="Techniques-for-Handling-the-Missing-Data"></a>

# Techniques for Handling the Missing Data

最好的防止数据丢失的方法就是更好地设计调研以及更仔细地收集数据。在临床研究中，足总和也列了六点在临床研究中关于防止数据丢失的基本建议。

在一项研究中，有一些数据丢失是在可容忍范围内的。对待数据丢失的一个方法是采用对数据丢失不敏感的数据分析方法。但是对数据丢失不敏感的分析方法不好找，下面的也是一些方法。

- Listwise or Case Deletion：最基本的方法就是不看丢失的数据，我们只看完整的数据。这个方法就叫 listwise deletion 了。不过当数据样本较小以及不符合 MCAR 时，这个方法不是很好。

- Pairwise Deletion：Pairwise deletion eliminates information only when the  
  particular data-point needed to test a particular assumption is missing.

- Mean Substitution：在这个方法中，用其他数据的平均值去替代丢失数据。

- Regression Imputation：对数据进行回归预测，然后用预测数据替代缺失数据。

- Last Observation Carried Forward：在长期数据收集中，可以采用最后一次的数据去代替丢失数据。

- Maximum Likelihood：就是用最大似然的方法去估计数据。

- Expectation-Maximization：这是一种最大似然的方法。

- Multiple Imputation：这个方法是说不想前面用一个值取替代缺失值，这里可以用一系列值去替代，因为预测的值可能并不只是一个。

- Sensitivity Analysis：Sensitivity analysis is defined as the study which defines how  
  the uncertainty in the output of a model can be allocated to the different sources of uncertainty in its inputs.

<a id="Recommendations"></a>

# Recommendations

这部分总体就是讲数据丢失是不可避免的，我们更应该多关注数据为什么丢失，然后去设计更好的方法去收集更完整的数据。
