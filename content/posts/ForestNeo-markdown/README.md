# ForestNeo 旧博客 Markdown 恢复

从 Hexo 发布目录 `ForestNeo-website-master` 恢复了 **120 篇文章**（2017—2023 年），其中 **32 篇带 DP / LDP 相关标签**。

文章按年份保存在 `articles/`，文件名使用原日期和原网址目录名，避免同名文章互相覆盖。每篇保留标题、发布日期、更新日期、分类、标签、作者和原网址。

正文恢复了标题层级、列表、引用、表格、链接、图片、代码和 LaTeX 公式。代码高亮的行号已去除；站内文章链接已指向本地恢复文件；原有标题锚点保留为少量 HTML，供目录和交叉链接跳转。

这是根据发布网页恢复的内容，原 Markdown 的空白、注释、排版选择及未发布内容无法从网页还原。正文中已有的笔误、重复标题、TODO 和未完成段落保留原样。远程图片保留原网址，未检查远程地址是否仍可访问。

## 校验

所有 120 篇均通过正文文本覆盖校验；409 处 MathJax 公式及 138 个代码块通过原文逐字校验，19 张表格、623 个图片引用及标题、列表数量核对通过。明细见 `recovery-report.json`。

首次恢复时额外把全部文章渲染为 HTML，核对了 6543 段正文文本、774 个原有锚点及 27 处本地跳转，全部通过。该次渲染校验及对应文件哈希保存在 `render-validation.json`；之后修改的文件可通过哈希识别。

## 文章清理

按要求删除了 33 篇旧文章，当前保留 **87 篇**。Python 文章仅保留两篇 bug 记录；原有新版博客文章不在本次旧文章清理范围内。`deleted-articles.json` 记录了删除清单，恢复脚本会跳过这些文章。下方恢复和校验数量为清理前记录。

已遍历全部文章，清除 **45 篇**文末的公众号宣传、二维码及其分隔线，其中包括两处宣传性质的“总结”和一处单独的二维码。正文及元数据保留，明细见 `footer-cleanup.json`；恢复脚本重新生成时也会自动清除这些结尾。

## 当前 Hugo 博客兼容性

本目录已放在 `content/posts/ForestNeo-markdown/`，当前保留的 87 篇文章均被 Hugo 的 `/posts/` 列表收录。现有 YAML 元数据中的标题、日期、分类和标签可以直接使用，`original_url` 等恢复信息可继续保留。

项目 `hugo.toml` 和根目录的 `layouts/` 已添加公式保留及渲染、Markdown 文章链接转换和本地图片路径处理。README、恢复脚本和 JSON 校验报告已排除出发布内容。部署脚本的 Hugo 版本已对齐本地验证版本 **0.147.9**，旧版 0.128 无法使用此次的公式渲染模板。

31 处旧文章地址（包括旧目录命名、`forestneo.topcom` 拼写和 `forestneo.com` 域名）已改为本地 Markdown 链接。Hugo 正式构建（`--gc --minify`）通过，全部 409 段恢复公式、138 个代码块、33 处站内跳转和 26 处本地图片引用检查通过；详见 `hugo-validation.json`。浏览器抽查了 Laplace、PCKV 和 Pandas 文章，公式和代码显示正常。上述本地图片仍是源网页中的占位图，原本缺失的图片无法通过路径适配恢复。

## 源文件中的图片问题

下列问题来自原网页，具体地址保存在校验报告中：

- [论文阅读-PROCHLO Strong Privacy for Analytics in the Crowd](<articles/2019/2019-11-05-PAPER-PROCHLO Strong Privacy for Analytics in theCrowd.md>)：1 张本地图片在源目录中缺失，原地址保留。
- [论文阅读-Federated Machine Learning Concept and Applications](<articles/2020/2020-03-14-PAPER-Federated Machine Learning Concept and Applications.md>)：1 张本地图片在源目录中缺失，原地址保留。
- [系统配置-macOS各种开发环境配置](<articles/2021/2021-11-09-系统配置-macOS各种开发环境配置.md>)：未转义的 configuration 标签转为行内代码。
- [密码学-不经意传输 Oblivious Transfer](<articles/2021/2021-11-20-技术扫盲-OT.md>)：13 张图片原本就是 1×1 内嵌占位图，已保存到 assets。
- [密码学-不经意传输 Oblivious Transfer](<articles/2021/2021-11-20-技术扫盲-OT-ForestNeo.md>)：13 张图片原本就是 1×1 内嵌占位图，已保存到 assets。

## DP / LDP 笔记

- [DP-Differential Privacy概念介绍](<articles/2018/2018-07-22-技术扫盲-Differential-Privacy概念介绍.md>)
- [DP-Laplace Mechanism](<articles/2018/2018-07-27-技术扫盲-DP-Laplace-Mechanism.md>)
- [DP-Exponential Mechanism](<articles/2018/2018-08-02-技术扫盲-DP指数机制.md>)
- [DP-Composition Theorem](<articles/2018/2018-08-04-技术扫盲-DP组合性质.md>)
- [论文阅读-Secure Two-Party Differentially Private Data Release for Vertically Partitioned Data](<articles/2018/2018-09-19-PAPER-Secure Two Party Differentially Private Data Release for Vertically Partitioned Data.md>)
- [论文阅读-Distance-Aware Encoding of Numerical Values for Privacy-Preserving Record Linkage](<articles/2018/2018-11-05-PAPER-Distance Aware Encoding of Numerical Values for Privacy Preserving Record Linkage.md>)
- [论文阅读-Privacy Preserving Triangle Counting in Large Graphs](<articles/2018/2018-11-05-PAPER-Privacy Preserving Triangle Counting in Large Graphs.md>)
- [论文阅读-RAPPOR：Randomized Aggregatable Privacy-Preserving Ordinal Response](<articles/2019/2019-01-02-PAPER-RAPPOR.md>)
- [论文阅读-Randomized Bit Vector：Privacy-Preserving Encoding Mechanism](<articles/2019/2019-01-08-PAPER-Randomized Bit Vector Privacy Preserving Encoding Mechanism.md>)
- [论文阅读-Collecting Telemetry Data Privately](<articles/2019/2019-01-26-PAPER-Collecting Telemetry Data Privately.md>)
- [论文阅读-Collecting and Analyzing Multidimensional Data with Local Differential Privacy](<articles/2019/2019-02-16-PAPER-Collecting and Analyzing Multidimensional Data with Local Differential Privacy.md>)
- [DP-Gaussian Mechanism](<articles/2019/2019-03-21-技术扫盲-Guassian-Mechanism.md>)
- [论文阅读-PrivKV Key-Value Data Collection with Local Differential Privacy](<articles/2019/2019-03-25-PAPER-PrivKV Key-Value Data Collection with Local Differential Privacy.md>)
- [论文阅读-A Utility-optimized Framework for Personalized Private Histogram Estimation](<articles/2019/2019-04-29-PAPER-A Utility-optimized Framework for Personalized Private Histogram Estimation.md>)
- [论文阅读-Private weighted histogram aggregation in crowdsourcing](<articles/2019/2019-07-04-PAPER-Private Weighted Histogram Aggregation in Crowdsourcing.md>)
- [论文阅读-Locally Differentially private Protocols for Frequency Estimation](<articles/2019/2019-07-17-PAPER-Locally Differentially Private Protocols for Frequency Estimation.md>)
- [论文阅读-The Staircase Mechanism in Differential Privacy](<articles/2019/2019-09-16-PAPER-The Staircase Mechanism in Differential Privacy.md>)
- [论文阅读-PROCHLO Strong Privacy for Analytics in the Crowd](<articles/2019/2019-11-05-PAPER-PROCHLO Strong Privacy for Analytics in theCrowd.md>)
- [论文阅读-Bidirectional Sampling for Handling Missing Data with Local Differential Privacy](<articles/2020/2020-02-25-PAPER-BiSample.md>)
- [论文阅读-PCKV: Locally Differentially Private Correlated Key-Value Data Collection with Optimized Utility](<articles/2020/2020-02-28-PAPER-PCKV Locally Differentially Private Correlated Key-Value Data Collection with Optimized Utility.md>)
- [论文阅读-FedSel Federated SGD under Local Differential Privacy with Top-k Dimension Selection](<articles/2020/2020-04-02-PAPER-FedSel Federated SGD under Local Differential Privacy with Top-k Dimension Selection.md>)
- [论文阅读-Hadamard Response](<articles/2020/2020-05-14-PAPER-Hadamard Response.md>)
- [论文阅读-TGM A Generative Mechanism for Publishing Trajectories with Differential Privacy](<articles/2020/2020-06-15-PAPER-TGM.md>)
- [论文阅读-Federated Learning of Deep Networks using Model Averaging](<articles/2021/2021-09-19-PAPER-Federated Learning of Deep Networks using Model Averaging.md>)
- [论文阅读-Towards Practical Differential Privacy for SQL Queries](<articles/2022/2022-01-27-PAPER-Towards Practical Differential Privacy for SQL Queries.md>)
- [DP-DP在企业的应用](<articles/2022/2022-02-05-技术扫盲-DP在企业的应用.md>)
- [DP-Composition Theorem（二）](<articles/2022/2022-02-07-技术扫盲-DP组合性质（二）.md>)
- [DP-震惊！美国人口普查局采用DP的真正原因（附资料）](<articles/2022/2022-02-18-技术扫盲-DP美国人口普查局采用DP的原因.md>)
- [论文阅读-Towards Practical Differential Privacy for SQL Queries](<articles/2022/2022-07-10-PAPER-AdaPDP.md>)
- [论文阅读-Differentially Private Federated Learning/ A Client Level Perspective](<articles/2022/2022-07-12-PAPER-Differentially Private Federated Learning A Client Level Perspective.md>)
- [论文阅读-Federated Learning With Differential Privacy Algorithms and Performance Analysis](<articles/2023/2023-04-24-PAPER-Federated Learning with Differential Privacy.md>)

## 全部文章

### 2017（12 篇）

- 2017-06-12 · [人生随笔-为什么鸡腿掉地上会让人很生气？](<articles/2017/2017-06-12-随笔-20170612-为什么鸡腿掉地上会让人很生气？.md>)
- 2017-08-28 · [人生随笔-我与审美](<articles/2017/2017-08-28-随笔-20170828-我与审美.md>)
- 2017-09-26 · [人生随笔-我是如何成为一个米粉的](<articles/2017/2017-09-26-随笔-20170926-我是如何成为一个米粉的.md>)
- 2017-10-01 · [人生随笔-国庆 & 中秋](<articles/2017/2017-10-01-随笔-20171001-国庆-中秋.md>)
- 2017-10-04 · [人生随笔-写在睡不着的夜晚](<articles/2017/2017-10-04-随笔-20171004-写在睡不着的夜晚.md>)
- 2017-10-05 · [人生随笔-写在华科里](<articles/2017/2017-10-05-随笔-20171005-写在华科里.md>)
- 2017-10-06 · [人生随笔-关于我（自黑篇）](<articles/2017/2017-10-06-随笔-20171006-关于我（自黑篇）.md>)
- 2017-11-27 · [人生随笔-像个段子一样活着](<articles/2017/2017-11-27-随笔-20171127-像个段子一样活着.md>)
- 2017-11-27 · [人生随笔-盘点那些年的活物](<articles/2017/2017-11-27-随笔-20171127-盘点那些年的活物.md>)
- 2017-12-11 · [人生随笔-怦然心动](<articles/2017/2017-12-11-随笔-20171211-怦然心动.md>)
- 2017-12-24 · [人生随笔-奕安的生日快乐](<articles/2017/2017-12-24-随笔-20171224-奕安的生日快乐.md>)
- 2017-12-31 · [人生随笔-那年我十八岁](<articles/2017/2017-12-31-随笔-20171231-那年我十八岁.md>)

### 2018（10 篇）

- 2018-01-07 · [人生随笔-这一份小心翼翼，给你](<articles/2018/2018-01-07-随笔-20180107-这一份小心翼翼.md>)
- 2018-04-08 · [人生随笔-怎样理解《残酷月光》](<articles/2018/2018-04-08-随笔-20180408-怎样理解《残酷月光》.md>)
- 2018-06-26 · [人生随笔-苦涩与生活](<articles/2018/2018-06-26-随笔-20180626-苦涩与生活.md>)
- 2018-07-22 · [DP-Differential Privacy概念介绍](<articles/2018/2018-07-22-技术扫盲-Differential-Privacy概念介绍.md>)
- 2018-07-27 · [DP-Laplace Mechanism](<articles/2018/2018-07-27-技术扫盲-DP-Laplace-Mechanism.md>)
- 2018-08-02 · [DP-Exponential Mechanism](<articles/2018/2018-08-02-技术扫盲-DP指数机制.md>)
- 2018-08-04 · [DP-Composition Theorem](<articles/2018/2018-08-04-技术扫盲-DP组合性质.md>)
- 2018-09-19 · [论文阅读-Secure Two-Party Differentially Private Data Release for Vertically Partitioned Data](<articles/2018/2018-09-19-PAPER-Secure Two Party Differentially Private Data Release for Vertically Partitioned Data.md>)
- 2018-11-05 · [论文阅读-Distance-Aware Encoding of Numerical Values for Privacy-Preserving Record Linkage](<articles/2018/2018-11-05-PAPER-Distance Aware Encoding of Numerical Values for Privacy Preserving Record Linkage.md>)
- 2018-11-05 · [论文阅读-Privacy Preserving Triangle Counting in Large Graphs](<articles/2018/2018-11-05-PAPER-Privacy Preserving Triangle Counting in Large Graphs.md>)

### 2019（20 篇）

- 2019-01-02 · [论文阅读-RAPPOR：Randomized Aggregatable Privacy-Preserving Ordinal Response](<articles/2019/2019-01-02-PAPER-RAPPOR.md>)
- 2019-01-05 · [《围城》摘抄](<articles/2019/2019-01-05-随笔-《围城》.md>)
- 2019-01-08 · [论文阅读-Randomized Bit Vector：Privacy-Preserving Encoding Mechanism](<articles/2019/2019-01-08-PAPER-Randomized Bit Vector Privacy Preserving Encoding Mechanism.md>)
- 2019-01-26 · [论文阅读-Collecting Telemetry Data Privately](<articles/2019/2019-01-26-PAPER-Collecting Telemetry Data Privately.md>)
- 2019-02-16 · [论文阅读-Collecting and Analyzing Multidimensional Data with Local Differential Privacy](<articles/2019/2019-02-16-PAPER-Collecting and Analyzing Multidimensional Data with Local Differential Privacy.md>)
- 2019-03-21 · [DP-Gaussian Mechanism](<articles/2019/2019-03-21-技术扫盲-Guassian-Mechanism.md>)
- 2019-03-25 · [论文阅读-PrivKV Key-Value Data Collection with Local Differential Privacy](<articles/2019/2019-03-25-PAPER-PrivKV Key-Value Data Collection with Local Differential Privacy.md>)
- 2019-03-25 · [人生随笔-收集的好句子](<articles/2019/2019-03-25-随笔-收集的好句子.md>)
- 2019-03-27 · [人生随笔-给你一个方程，与那一时自觉聪明](<articles/2019/2019-03-27-随笔-20190327-给你一个方程，与那一时自觉聪明.md>)
- 2019-04-29 · [论文阅读-A Utility-optimized Framework for Personalized Private Histogram Estimation](<articles/2019/2019-04-29-PAPER-A Utility-optimized Framework for Personalized Private Histogram Estimation.md>)
- 2019-06-17 · [论文阅读-Approximate multiple count in Wireless Sensor Networks](<articles/2019/2019-06-17-PAPER-Approximate multiple count in Wireless Sensor Networks.md>)
- 2019-07-03 · [人生随笔--小时候](<articles/2019/2019-07-03-随笔-20190703-小时候.md>)
- 2019-07-04 · [论文阅读-Private weighted histogram aggregation in crowdsourcing](<articles/2019/2019-07-04-PAPER-Private Weighted Histogram Aggregation in Crowdsourcing.md>)
- 2019-07-17 · [论文阅读-Locally Differentially private Protocols for Frequency Estimation](<articles/2019/2019-07-17-PAPER-Locally Differentially Private Protocols for Frequency Estimation.md>)
- 2019-09-16 · [论文阅读-The Staircase Mechanism in Differential Privacy](<articles/2019/2019-09-16-PAPER-The Staircase Mechanism in Differential Privacy.md>)
- 2019-10-09 · [数学基础-皮尔森相关系数 (Pearson Correlation Coefficient)](<articles/2019/2019-10-09-数学基础-皮尔森相关系数-Pearson-Correlation-Coefficient.md>)
- 2019-11-05 · [论文阅读-PROCHLO Strong Privacy for Analytics in the Crowd](<articles/2019/2019-11-05-PAPER-PROCHLO Strong Privacy for Analytics in theCrowd.md>)
- 2019-11-06 · [密码学-Secret Sharing](<articles/2019/2019-11-06-技术扫盲-Secret-Sharing.md>)
- 2019-11-11 · [论文阅读-The Prevention and Handling of the Missing Data](<articles/2019/2019-11-11-PAPER-The Prevention and Handling of the Missing Data.md>)
- 2019-12-25 · [论文阅读-Resolving Conflicts in Heterogeneous Data by Truth Discovery and Source Reliability Estimation](<articles/2019/2019-12-25-PAPER-Resolving Conflicts in Heterogeneous Data by Truth Discovery and Source Reliability Estimation.md>)

### 2020（21 篇）

- 2020-02-25 · [论文阅读-Bidirectional Sampling for Handling Missing Data with Local Differential Privacy](<articles/2020/2020-02-25-PAPER-BiSample.md>)
- 2020-02-26 · [吴恩达机器学习课程(4)-Linear Regression with Multiple Variables](<articles/2020/2020-02-26-ML-吴恩达机器学习课程-04-Linear Regression with Multiple Variables.md>)
- 2020-02-27 · [吴恩达机器学习课程(6)-Logistic Regression](<articles/2020/2020-02-27-ML-吴恩达机器学习课程-06-Logistic Regression.md>)
- 2020-02-27 · [吴恩达机器学习课程(7)-Regularization](<articles/2020/2020-02-27-ML-吴恩达机器学习课程-07-Regularization.md>)
- 2020-02-28 · [吴恩达机器学习课程(8)-Neural Networks:Representation](<articles/2020/2020-02-28-ML-吴恩达机器学习课程-08-Neural Networks Representation.md>)
- 2020-02-28 · [吴恩达机器学习课程(9)-Neural Networks:Learning](<articles/2020/2020-02-28-ML-吴恩达机器学习课程-09-Neural Networks Learning.md>)
- 2020-02-28 · [论文阅读-PCKV: Locally Differentially Private Correlated Key-Value Data Collection with Optimized Utility](<articles/2020/2020-02-28-PAPER-PCKV Locally Differentially Private Correlated Key-Value Data Collection with Optimized Utility.md>)
- 2020-03-04 · [开源项目-sunDP介绍](<articles/2020/2020-03-04-开源项目-sunDP介绍.md>)
- 2020-03-11 · [公开课学习-分布式机器学习（上）](<articles/2020/2020-03-11-视频资源-分布式机器学习（上）.md>)
- 2020-03-12 · [公开课学习-分布式机器学习（中）](<articles/2020/2020-03-12-视频资源-分布式机器学习（中）.md>)
- 2020-03-13 · [公开课学习-分布式机器学习（下）](<articles/2020/2020-03-13-视频资源-分布式机器学习（下）.md>)
- 2020-03-14 · [论文阅读-Federated Machine Learning Concept and Applications](<articles/2020/2020-03-14-PAPER-Federated Machine Learning Concept and Applications.md>)
- 2020-04-02 · [论文阅读-FedSel Federated SGD under Local Differential Privacy with Top-k Dimension Selection](<articles/2020/2020-04-02-PAPER-FedSel Federated SGD under Local Differential Privacy with Top-k Dimension Selection.md>)
- 2020-04-20 · [吴恩达机器学习课程(10)-Advice for Applying Machine Learning](<articles/2020/2020-04-20-ML-吴恩达机器学习课程-10-Advice for Applying Machine Learning.md>)
- 2020-04-20 · [吴恩达机器学习课程(11)-Machine Learning System Design](<articles/2020/2020-04-20-ML-吴恩达机器学习课程-11-Machine Learning System Design.md>)
- 2020-04-20 · [吴恩达机器学习课程(12)-Support Vector Machines](<articles/2020/2020-04-20-ML-吴恩达机器学习课程-12-Support Vector Machines.md>)
- 2020-04-20 · [吴恩达机器学习课程(13)-Clustering](<articles/2020/2020-04-20-ML-吴恩达机器学习课程-13-Clustering.md>)
- 2020-04-21 · [吴恩达机器学习课程(14)-Dimensionality Reduction](<articles/2020/2020-04-21-ML-吴恩达机器学习课程-14-Dimensionality Reduction.md>)
- 2020-05-07 · [论文阅读-Federated Learning in Mobile Edge Networks A Comprehensive Survey](<articles/2020/2020-05-07-PAPER-Federated Learning in Mobile Edge Networks A Comprehensive Survey.md>)
- 2020-05-14 · [论文阅读-Hadamard Response](<articles/2020/2020-05-14-PAPER-Hadamard Response.md>)
- 2020-06-15 · [论文阅读-TGM A Generative Mechanism for Publishing Trajectories with Differential Privacy](<articles/2020/2020-06-15-PAPER-TGM.md>)

### 2021（14 篇）

- 2021-08-30 · [周志华《机器学习》-02-模型评估与选择](<articles/2021/2021-08-30-ML-周志华《机器学习》-02-模型评估与选择.md>)
- 2021-08-31 · [人生随笔-这就是老一辈的爱情吧](<articles/2021/2021-08-31-随笔-20200831-这就是老一辈的爱情吧.md>)
- 2021-09-13 · [技术扫盲-Private Set Integration](<articles/2021/2021-09-13-技术扫盲-Private Set Integration.md>)
- 2021-09-19 · [论文阅读-Federated Learning of Deep Networks using Model Averaging](<articles/2021/2021-09-19-PAPER-Federated Learning of Deep Networks using Model Averaging.md>)
- 2021-10-12 · [技术扫盲-一文带你读懂纵向联邦学习中的线性回归模型](<articles/2021/2021-10-12-技术扫盲-一文带你读懂纵向联邦学习的线性回归问题.md>)
- 2021-11-01 · [系统配置-软件推荐](<articles/2021/2021-11-01-系统配置-软件推荐.md>)
- 2021-11-02 · [论文阅读-How to Backdoor Federated Learning](<articles/2021/2021-11-02-PAPER-How to Backdoor Federated Learning.md>)
- 2021-11-02 · [系统配置-MdNice开发配置](<articles/2021/2021-11-02-系统配置-MdNice开发配置.md>)
- 2021-11-09 · [系统配置-macOS各种开发环境配置](<articles/2021/2021-11-09-系统配置-macOS各种开发环境配置.md>)
- 2021-11-11 · [人生随笔-我的小米产品](<articles/2021/2021-11-11-随笔-我的小米产品.md>)
- 2021-11-20 · [密码学-不经意传输 Oblivious Transfer](<articles/2021/2021-11-20-技术扫盲-OT.md>)
- 2021-11-20 · [密码学-不经意传输 Oblivious Transfer](<articles/2021/2021-11-20-技术扫盲-OT-ForestNeo.md>)
- 2021-11-29 · [技术扫盲-树模型](<articles/2021/2021-11-29-技术扫盲-树模型.md>)
- 2021-12-06 · [论文阅读-Practical Secure Aggregation for Privacy-Preserving Machine Learning](<articles/2021/2021-12-06-PAPER-Practical Secure Aggregation for Privacy-Preserving Machine Learning.md>)

### 2022（9 篇）

- 2022-01-11 · [论文阅读-SecureBoost-A Lossless Federated Learning Framework](<articles/2022/2022-01-11-PAPER-SecureBoost.md>)
- 2022-01-27 · [论文阅读-Towards Practical Differential Privacy for SQL Queries](<articles/2022/2022-01-27-PAPER-Towards Practical Differential Privacy for SQL Queries.md>)
- 2022-02-05 · [DP-DP在企业的应用](<articles/2022/2022-02-05-技术扫盲-DP在企业的应用.md>)
- 2022-02-07 · [DP-Composition Theorem（二）](<articles/2022/2022-02-07-技术扫盲-DP组合性质（二）.md>)
- 2022-02-18 · [DP-震惊！美国人口普查局采用DP的真正原因（附资料）](<articles/2022/2022-02-18-技术扫盲-DP美国人口普查局采用DP的原因.md>)
- 2022-07-10 · [论文阅读-Towards Practical Differential Privacy for SQL Queries](<articles/2022/2022-07-10-PAPER-AdaPDP.md>)
- 2022-07-12 · [论文阅读-Differentially Private Federated Learning/ A Client Level Perspective](<articles/2022/2022-07-12-PAPER-Differentially Private Federated Learning A Client Level Perspective.md>)
- 2022-08-15 · [Python-Pyspark中UDF与random同时使用时bug记录F.md](<articles/2022/2022-08-15-编程-Python-Pyspark中UDF与random同时使用时bug记录.md>)
- 2022-09-11 · [Python-一次调试装饰器中的bug记录](<articles/2022/2022-09-11-编程-Python-一次调试装饰器中的bug记录.md>)

### 2023（1 篇）

- 2023-04-24 · [论文阅读-Federated Learning With Differential Privacy Algorithms and Performance Analysis](<articles/2023/2023-04-24-PAPER-Federated Learning with Differential Privacy.md>)

## 重新生成

`recover.py` 需要 Python 和 lxml。可使用本机 Codex 自带的 Python，从项目根目录执行：

```sh
/Users/sunlin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 content/posts/ForestNeo-markdown/recover.py --overwrite
```

该命令只重新生成恢复目录中的文章和报告，原 HTML 不会被修改。
