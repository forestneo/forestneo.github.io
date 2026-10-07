---
title: "论文阅读-Towards Practical Differential Privacy for SQL Queries"
date: "2022-01-27T19:14:04+08:00"
updated: "2022-01-27T19:14:04+08:00"
categories: ["论文阅读"]
tags: ["SQL", "DP"]
author: "DPer"
original_url: "http://forestneo.top/2022/01/27/PAPER-Towards Practical Differential Privacy for SQL Queries/"
source_html: "ForestNeo-website-master/2022/01/27/PAPER-Towards Practical Differential Privacy for SQL Queries/index.html"
mathjax: true
---

文章地址：  
标题：Towards Practical Differential Privacy for SQL Queries  
作者：Joah Johnson, Joseph P. Near, Dawn Song

先简单梳理一下本文总的脉络吧，论文结构分为9个章节，如下：

- 0 Abstract

- 1 Introduction

- 2 Requirements for Practical Differenital Privacy

- 3 Elastic Sensitivity

- 4 FLEX: Practical Differential Privacy for SQL Queries

- 5 Experimental Evaluation

- 6 Related Work

- 7 Conclusion

<a id="Elastic-Sensitivity"></a>

# Elastic Sensitivity

这个东西，本文称为弹性敏感度。

<a id="Definition-of-Elastic-Sensitivity"></a>

## Definition of Elastic Sensitivity

一个查询的弹性敏感度根据查询的结构递归给出。为了支持smoothing functions，本文的定义描述如何计算 $k$ 距离内的弹性敏感度（在这个定义下，local sensitivity 相当于 $k=0$）。

![image-20220127113255770](<https://forest-pic.oss-cn-beijing.aliyuncs.com/webimg/202201271133555.png>)

图1展示了完整的定义，由四部分组成：

- Core relational algebra

- Definition of elastic sensitivity

- Max frequency at distance $k$

- Ancestors of a relation

后面开始分别对这四部分进行介绍。

**Core relational algebra.** 本文只对以下四种关系代数进行处理，包含：selection ($\sigma$)、projection ($\pi$)、join ($\bowtie$)、countiong ($Count$) 和对分组的counting($Count_{G_1, …, G_n}$)。同时本文允许 equijoins 等值连接，包括 self-join, 和其他的join（一对一，一对多和多对多）。

出于 presentation 的目的，本文假设每个 query 的最外层都是 count 操作。然而，只要查询不对聚合结果执行算术或其他修改，该方法可以扩展到嵌套在查询中任何位置的聚合。 例如，以下查询计算总行程数并投影“count”属性：

$$
\pi_{\operatorname{count}} Count(\operatorname{trips})
$$

本文方法也可以支持这种请求。

**Elastic Sensitivity.** 图1(b) 展示了递归定义距离 $k$ 上的弹性敏感度。数据集 $x$上请求 $q$ 的 $k$ 距离的弹性敏感度用 $\hat{S}^{(k)}(q, x)$ 表示，函数 $\hat{S}$ 由变化的弹性稳定性（elastic stability）定义，记为 $\hat{S}_R$。

<a id="Optimization-for-Public-Tables"></a>

## Optimization for Public Tables

弹性敏感度（Elastic Sensitivity）的定义中，本文假设数据表中所有记录都必须受到保护。在实践中，数据库通常包含敏感和非敏感数据， 这可以用来加强我们对连接非敏感表的查询的 Local Sensitivity 的限制。例如，在我们的数据集中，城市数据是公开的，因此系统不需要防止攻击者学习有关城市表的信息。当然，需要注意，公共表集是特定于域的，并且在每个数据环境中都会有所不同。

更准确地说，在连接表达式 `T1 JOIN T2 ON T1.A = T2.B` 中，如果 `T2` 是公开的，则连接的 Elastic Stability 等于 `T1` 的 Elastic Stability 乘以 T2.B 的最大频率。此公式可防止使用具有重复连接键的公共表泄露有关私有表的信息。

<a id="FLEX-Practical-Differential-Privacy-for-SQL-Queries"></a>

# FLEX: Practical Differential Privacy for SQL Queries

<a id="Effciently-Calculating-S"></a>

## Effciently Calculating $S$
