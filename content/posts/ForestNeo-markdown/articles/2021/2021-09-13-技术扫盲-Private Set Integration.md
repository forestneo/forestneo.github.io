---
title: "技术扫盲-Private Set Integration"
date: "2021-09-13T00:00:00+08:00"
updated: "2022-01-05T11:47:11+08:00"
categories: ["技术扫盲", "Private Set Integration"]
tags: ["Private Set Integration"]
author: "DPer"
original_url: "http://forestneo.top/2021/09/13/技术扫盲-Private Set Integration/"
source_html: "ForestNeo-website-master/2021/09/13/技术扫盲-Private Set Integration/index.html"
mathjax: true
---

<a id="PSI-基本概念"></a>

# PSI 基本概念

PSI 全称 Private Set Intersection，翻译为隐私集合求交。说白点就是在多个实体（数据集）间求交集但是又要保证隐私。

![image-20210908203405238](<https://forest-pic.oss-cn-beijing.aliyuncs.com/webimg/202109082034281.png>)

介绍PSI简化一点，就是有集合A和集合B，求集合$A \cap B$。在概念上，A和B的地位是平等的，但是通常在协议执行的过程中，PSI 的结果通常先由一方获得，然后再同步给另一方。

讨论安全方案的时候，一般有个安全模型。通常分为半可信模型和恶意模型去讨论：

- 半可信模型（Semi-honest model）：通常也称为HBC（Honest but Curious）模型，即参与协议的实体会遵守协议内容，但是他将对数据保持好奇，可以尝试对数据进行观察，反推等。

- 恶意模型（Malicious model）：参与方是恶意的，他可能修改协议结果，或者破坏协议运行等方式以节约算力或窃取对方数据。

<a id="PSI方法介绍"></a>

# PSI方法介绍

在应用落地阶段，通常不会考虑含有第三方的方案。下面介绍以下几种PSI方法：

- 不安全的 PSI 方案：基于哈希的方案

- 安全的PSI方案：

  - 基于 Diffie-Hellmann 的 PSI 方案

  - 基于 RSA 的 PSI 方案

  - 基于不经意传输的方案

<a id="基于哈希的PSI"></a>

## 基于哈希的PSI

对于哈希函数 $H$，给定 $H(x), H(y)$，若 $H(x)=H(y)$，则我们可以认为 $x=y$。因此这个原理可以用来做PSI了。

首先AB共同协商一个哈希函数用于计算哈希值，比如MD5，然后A把每条记录的哈希值发送给B，B来计算交集。这个过程十分有效并且易于理解，但是当输入域较小的时候可能会遭受到膨胀攻击。不过如果输入集合的取值范围非常大的时候，这个方案也比较安全有效。但是很可惜，如此大的输入域在现实应用场景中不一定是存在的。

<a id="基于-Diffie-Hellmann-的-PSI-方案"></a>

## 基于 Diffie-Hellmann 的 PSI 方案

此方案可追溯到1986年，是学术界最早的PSI方案，可分为基于 Diffie-Hellmann的PSI方案与基于RSA盲签名的PSI方案，接下来以此对二者进行介绍。

假设A拥有数据 $x$，B拥有数据 $y$，然后双方协商 $N$ 是个大素数，并且出于安全角度考虑，$(N-1)/2$也要是个大素数。然后所有的运算结果都需要对 $N$ 取模。然后A产生私钥 $\alpha$，B产生私钥$\beta$。那么如果$x=y$，则有$(x^\alpha)^\beta=(y^\beta)^\alpha$，反之亦然。因此可以用这个原理判断两条数据是否相等并进行PSI求解。

因此可以这么求解这个过程：

1. A计算 $x^\alpha \pmod N$，发送给B，B计算 $y^\beta \pmod N$ 发送给A；

2. B拿到 $x^\alpha$，计算 $(x^\alpha)^\beta$ 并返回给A，此时，A计算 $(y^\beta)^\alpha$ 并返回给B；

3. 双方根据 $(x^\alpha)^\beta$与$(y^\beta)^\alpha$ 是否相等判断 $x,y$ 是否相等。

在具体应用的过程中，我们不会直接用$x,y$进行计算，而是先对$x,y$进行一个$H(\cdot)$函数的计算，可以理解为求哈希。学术上这个$H$函数叫做 random oracle。

<a id="基于RSA的PSI方案"></a>

## 基于RSA的PSI方案

<a id="协议介绍"></a>

### 协议介绍

这个方案在论文《Practical Private Set Intersection Protocols with Linear Computational and Bandwidth Complexity》中有提到，如下图所示：

![image-20210928174127873](<https://forest-pic.oss-cn-beijing.aliyuncs.com/webimg/202110082202762.png>)

一开始看的时候，始终没搞明白 $H’$ 是啥玩意，网上的介绍直接跳过了，原来也没啥，就是个哈希函数，原文的定义为：

- $H()$：cryptographic hash function, codomain depends on context

- $H’()$：cryptographic hash function $H’:\{0,1\}^{*} \rightarrow\{0,1\}^\tau$

我们假设有两个实体，分别是 Client 和 Server。首先有这么几个前期准备：

1. Client 的数据表示为 $c_1, c_2, …, c_v$，Server 的数据表示为 $s_1, s_2, …, s_w$，同时，其对应的哈希之后的结果分别是 $hc_i = H(c_i), hs_j=H(s_j)$；

2. Server 产生 RSA 的秘钥，自己拿私钥$(d,n)$，公钥 $(e,n)$ 给到 Client；

3. 准备一个 Full-Domain Hash 函数 $H(\cdot)$；

4. Client 选取随机数 $R_c$，要满足 $(R_c, n)=1$。

现在开始工作了，我们一步一步来看：

1. Server 对数据加密，并且对加密的结果进行签名：

   $$
   K_{s:j}=(hs_j)^d \mod n, \quad t_j = H'(K_{s:j})
   $$

2. Client 产生随机数 $R_c$，对数据进行公钥加密，并且乘以 $R_c$ 进行加盲扰动(每条数据都对应一个不同的 $R_c$)：

   $$
   y_i = hc_i \cdot (R_{c:i})^e \mod n
   $$

3. Client 把 $\{y_1, y_2, …, y_v\}$ 发送给 Server；

4. Server 进行如下计算：

   $$
   \forall i, \quad y_i' = (y_i)^d \mod n
   $$

5. Server 把 $\{y_1’,…,y_v’\}, \{t_1, …, t_2\}$ 发送给 Client；

6. Client 进行如下计算：

   $$
   \forall i, \quad K_{c:i} = y_i'/R_{c:i}, \quad t_i'=H'(K_{c:i})
   $$

7. 返回 $\{t_1’,…,t_v’\} \cap\{t_1…,t_w\}$

<a id="问题分析"></a>

### 问题分析

我们来重复分析一下这个过程为什么要这么设计，本质问题就是 Client 有一个 $c$，Server 有一个 $s$，如何判断是否满足 $c=s$，两边都哈希一下的话，就是如何判断 $hc=hs$。但是吧，我们又不能泄露 $hc,hs$，所以我们转过去比较 $hc^d=hs^d$，然后或许是出于安全考虑吧，我们再搞一个哈希函数，来比较 $H’(hc^d)=H’(hs^d)$。对于 Server 来说，他有秘钥 $d$，所以很容易计算出 $H’(hs^d)$，并且这个东西是可以公开，然后重点就变成了 Client 怎么算 $H’(hc^d)$，因为 $H’$ 可以后来再算，所以问题等价于怎么算 $hc^d$，而尴尬的是 Client 他只有 $e$，没有 $d$。接下来看看怎么办。

首先 Client 搞个随机数 $R$，并且算一个 $hc\cdot R^e$（为了说明问题方便，$\mod n$ 忽略了），把这玩意发给 Server，Server 就算 $(hc\cdot R^e)d=hc^d \cdot R$，然后这个东西返还给 Client 之后，Client 除以一个 $R$，就得到了 $hc^d$ 了。所以这个流程就 ok 了。

<a id="适用场景"></a>

### 适用场景

这个方案适合两方数据量差别很大的场景，可以拥有数量多的用户拿私钥 $d$，即 Client 数量多一点，Server 数量少一点比较好。

<a id="其它问题"></a>

### 其它问题

> 既然哈希函数已经是不可逆的，那么为什么还要搞这么一套？

哈希函数的不可逆是对定义域内的所有数据而言的，而在应用阶段，用来做 PSI 的数据通常是手机号，身份证ID 等关键信息，容易被暴力穷举的。

> 每个数据是不是要对应不同的 $R$？

是的，不然可以通过多个数据和对应密文反推出 $R$。

<a id="参考资料"></a>

### 参考资料

- [诸葛蜗牛 - 知乎 (zhihu.com)](<https://www.zhihu.com/people/dbkcpp>)

- [纵向联邦学习里的加密样本ID对齐 - 知乎 (zhihu.com)](<https://zhuanlan.zhihu.com/p/138557549>)

- Practical Private Set Intersection Protocols with Linear Computational and Bandwidth Complexity

<a id="基于不经意传输的-PSI-方案"></a>

## 基于不经意传输的 PSI 方案

（先占个坑）

<a id="PSI用途"></a>

# PSI用途

我在调研PSI的时候，用途主要是在联邦学习中的数据对齐。在纵向联邦学习中，不同合作方拥有的数据具备不同的字段，比如医院和保险公司合作，他们可以按照身份证ID对数据进行对齐。这个时候就需要在不泄露身份证的前提下去做PSI。

参考

- [隐私集合求交（Private Set Intersection，PSI）技术的发展、挑战与未来 （一） - 知乎 (zhihu.com)](<https://zhuanlan.zhihu.com/p/396327172>)

- [隐私集合求交（Private Set Intersection，PSI）技术的发展、挑战与未来 （二） - 知乎 (zhihu.com)](<https://zhuanlan.zhihu.com/p/407290294>)

- [纵向联邦学习里的加密样本ID对齐](<https://zhuanlan.zhihu.com/p/138557549>)
