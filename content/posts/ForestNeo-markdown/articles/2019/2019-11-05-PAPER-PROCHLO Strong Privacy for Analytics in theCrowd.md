---
title: "论文阅读-PROCHLO Strong Privacy for Analytics in the Crowd"
date: "2019-11-05T11:06:08+08:00"
updated: "2021-10-10T20:31:47+08:00"
categories: ["论文阅读"]
tags: ["Differential Privacy"]
author: "DPer"
original_url: "http://forestneo.top/2019/11/05/PAPER-PROCHLO Strong Privacy for Analytics in theCrowd/"
source_html: "ForestNeo-website-master/2019/11/05/PAPER-PROCHLO Strong Privacy for Analytics in theCrowd/index.html"
---

<a id="Introduction"></a>

# Introduction

<a id="Motivation-and-Alternatives"></a>

# Motivation and Alternatives

<a id="The-Encode-Shuffle-Analyze-Architecture"></a>

# The Encode-Shuffle-Analyze Architecture

![](<http://forestneo.top/2019/11/05/PAPER-PROCHLO Strong Privacy for Analytics in theCrowd/fig01.png>)

ESA 架构主要分为三层，分别是：Encoders，Shufflers 以及 Analyzers，就像上面的这个图一样。

Encoders 运行在用户的客户端设备。其对监测的数据进行一定的保护。例如采用嵌套的加密方法来保护哪个实体拥有哪些数据以及数据的顺序。出于隐私需求，encoders 还可以对客户端数据进行一定的处理，比如加上随机噪声。

Shuffler 是个独立的网络服务提供方，他从 Encoders 端接受加密的编码数据。它的功能是对数据进行隐藏，防止数据可以溯源（去掉报文到达时间，IP地址信息等），同时对数据进行重新排序以防止 thwart traffic analysis。

<a id="Keys-Trust-and-Attack-Models"></a>

## Keys, Trust, and Attack Models

<a id="Encoder"></a>

## Encoder

<a id="Shuffler"></a>

## Shuffler

从逻辑上说，Shuffler 将用户端和分析端隔离，提供了一个隐私保护层。

Shuffler 主要完成四个任务：anonymization, shuffling, thresholding 以及 batching。

- **anonymization**：Shuffler 接收到从 Encoder 发来的数据，因此 Shuffler 是可以从报文中访问到和用户有关的 metadata 的，如 timestamps, source IP addresses, routing paths 等。因此 shuffler 的一个重要功能就是提供匿名化支持，shuffler 需要将这一类隐含用户信息的 metadata 剥离出来。匿名化的核心目标就是使得 user 和 report 相互独立。

- **shuffling**：那么匿名化能否做到使得 user 和 report 相互独立呢？其实不是的，在特定攻击模型中，analyzer 可能会监控网络流量，通过对比 shuffler 接收到报文的时间和 shuffler 将报文发送给自己的时间就知道哪个 report 是 哪个 user 的。因此 shuffler 并不是接收到一个 report 就进行 anonymization 并转发。而是**将接收到的 reports 打包，并且随意置换顺序，然后以低频率方式统一发送。**

- **thresholding**：

- **batching**：

<a id="Analyzer"></a>

## Analyzer

<a id="Privacy-Guarantees"></a>

## Privacy Guarantees

<a id="PROCHLO-Implementation-and-Hardening"></a>

# PROCHLO Implementation and Hardening

<a id="Higher-Assurance-by-using-Trustworthy-Hardware"></a>

## Higher Assurance by using Trustworthy Hardware

<a id="SGX-Attestation-for-Networked-Shuffler-Services"></a>

### SGX Attestation for Networked Shuffler Services

<a id="Oblivious-Shuffling-within-SGX-Enclaves"></a>

### Oblivious Shuffling within SGX Enclaves

![fig02](<https://forest-pic.oss-cn-beijing.aliyuncs.com/fig02.png>)

<a id="State-of-the-Art-in-Oblivious-Shuffling"></a>

### State of the Art in Oblivious Shuffling

作者想要找这样一个 oblivious-shuffling 算法，其要满足密码学意义上的安全同时要有好的扩展性和效率。结果是没找到，所以他们自己设计了一个叫做 Stash Shuffle 的算法。

<a id="The-Stash-Shuffle-Algorithm"></a>

### The Stash Shuffle Algorithm

由于已有的 oblivious shuffling 算法并不满足scalability, efficiency or randomness of the permutations，作者们设计了一个叫做 Stash Shuffle 的算法，这个算法可以在这里单独看到：[Oblivious Stash Shuffle](<https://arxiv.org/pdf/1709.07553>)。

的

<a id="Crowd-Cardinality-Thresholding-inside-SGX"></a>

### Crowd Cardinality Thresholding inside SGX

<a id="Encoding-Using-Secret-Sharing-Cryptography"></a>

## Encoding Using Secret-Sharing Cryptography

<a id="Blinded-Crowd-IDs-for-Shuffler-Thresholding"></a>

## Blinded Crowd IDs for Shuffler Thresholding

<a id="Implementation"></a>

## Implementation

<a id="Evaluation"></a>

# Evaluation

<a id="Stash-Shuffle"></a>

## Stash Shuffle

<a id="Vocab-Empirical-Long-tail-Distributions"></a>

## Vocab: Empirical Long-tail Distributions

<a id="Perms-User-Actions-Regarding-Permissions"></a>

## Perms: User Actions Regarding Permissions

<a id="Suggest-Predicting-the-Next-Contend-Viewed"></a>

## Suggest: Predicting the Next Contend Viewed

<a id="Flix-Collaborative-Filtering"></a>

## Flix: Collaborative Filtering

<a id="Conclusions"></a>

# Conclusions

![fig01](<https://forest-pic.oss-cn-beijing.aliyuncs.com/fig01.png>)

![fig03](<https://forest-pic.oss-cn-beijing.aliyuncs.com/fig03.png>)

![fig04](<https://forest-pic.oss-cn-beijing.aliyuncs.com/fig04.png>)

![fig05](<https://forest-pic.oss-cn-beijing.aliyuncs.com/fig05.png>)

![alg01](<https://forest-pic.oss-cn-beijing.aliyuncs.com/alg01.png>)

![alg02](<https://forest-pic.oss-cn-beijing.aliyuncs.com/alg02.png>)

![alg03](<https://forest-pic.oss-cn-beijing.aliyuncs.com/alg03.png>)

![alg04](<https://forest-pic.oss-cn-beijing.aliyuncs.com/alg04.png>)

![tbl01](<https://forest-pic.oss-cn-beijing.aliyuncs.com/tbl01.png>)

![tbl02](<https://forest-pic.oss-cn-beijing.aliyuncs.com/tbl02.png>)

![tbl03](<https://forest-pic.oss-cn-beijing.aliyuncs.com/tbl03.png>)

![tbl04](<https://forest-pic.oss-cn-beijing.aliyuncs.com/tbl04.png>)

![tbl05](<https://forest-pic.oss-cn-beijing.aliyuncs.com/tbl05.png>)
