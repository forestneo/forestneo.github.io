---
title: "密码学-不经意传输 Oblivious Transfer"
date: "2021-11-20T00:00:00+08:00"
updated: "2021-12-15T14:08:33+08:00"
categories: ["技术扫盲", "密码学"]
tags: ["OT"]
author: "DPer"
original_url: "http://forestneo.top/2021/11/20/技术扫盲-OT-ForestNeo/"
source_html: "ForestNeo-website-master/2021/11/20/技术扫盲-OT-ForestNeo/index.html"
---

> 本文内容转载自公众号《陆叁》，作者井渭展，本科毕业于南京航空航天大学信息安全专业，目前在中国科学院信息工程研究所攻读硕士学位。研究兴趣包括隐私保护、安全多方计算、同态加密和机器学习。知乎：基因井

不经意传输（OT，oblivious transfer）是一个密码学协议，目前被广泛的应用于安全多方计算（SMPC，Secure Multi-Party Computation）。它由 Rabin<sup><a href="#fn_Rabin" id="reffn_Rabin">Rabin</a></sup>在 1981 年提出。本文梳理总结了1981年到2013年之间，不经意传输协议的发展脉络，并总结了关键技术。

一、Rabin 1981 提出

它是为了解决如下的问题而产生：Alice 拥有秘密 ，Bob 拥有秘密 。Alice 和 Bob 想要交换秘密，要求两方都有可能得到秘密并且秘密拥有方不知道对方是否得到秘密。具体方案如下：

> 方案假设两方的秘密都是单比特!!  
> (1) 随机选取两个大素数 、 。并计算得到 ，然后将 发送给 。  
> (2) 随机选取一个数 ，要求 ，计算 ，然后将 和私钥加密的 发送给 。  
> (3) 找到一个 使得 ，发送 给 。  
> (4) 计算 ，此时有 。  
> (5) 根据下面公式计算
>
> 接着计算 然后将 发送给

这是 Alice 获得 Bob 的秘密 的过程，可以得出 Alice 得到 的概率为 。Bob获得 的过程依然是上述步骤，只不过是将 Alice 和 Bob 角色互换。

Rabin 提出的方案两方都无法获得对方的秘密的概率是 ，可以成功交换的概率是 。可见该方案还不是很完善，不能保证两方每次都能在满足要求的情况下获得秘密，还不具有应用意义。所以有了 1985 年 Even\[2\]等人在此\[1\]基础上提出的新的 1-out-2 OT 协议\[2\]。

1985 1-out-of-2 OT

Even 等人的提出新的使用公钥密码体制的 1-out-of-2 OT 协议，给出了 OT 公理化的定义和实现。相比于 Rabin 等人提出的一方只有 的概率获得秘密，Even等人将其进行了改进，即：Alice 拥有两个秘密 ，而 Bob 想要知道其中一个。在 OT 协议执行完成之后，Bob 获得了其中一个秘密，但是不知道另外一条秘密 ，并且 Alice 也不知道 Bob 选择的是 ，还是 。如下图：

![图片](<../../assets/44d5041f007d30a9.png>)

具体过程为：

> (1) 在公钥密码中随机选择一组密钥 ;  
> 同时随机选择加密算32法明文空间R的两个明文 ;  
> 最后将 和 发送给 .  
> (2) 随机选择 ;  
> 然后在与(1)同样的明文空间选择 ;  
> 计算 ,并且向 发送 ;  
> (3) 计算 ,其中 ;  
>  随机选择变量 ;  
> 进而将三元组 发送给 .  
> (4) 根据自己选择 以及三元组最后一项 来选择第一项或者第二项；  
>  相同选第一项，不同选第二项，与自己 进行 运算得到秘密 . 是模运算加法， 是模运算减法。模运算的 就是明文空间大小

可以清楚的看到，1-out-of-2 OT 执行结束之后，Bob获得了一个秘密且不知到另外一条秘密，而 Alice 则不知到 Bob 拿到了哪一条秘密。1-out-of-2 OT 是一个具有实际应用意义的不经意传输协议，也是目前较为常用的一种。

1986 1-out-of-n OT

之后在1986年，Brassard\[3\] 等人继续将OT协议改进到了 1-out-of-n OT 版本。与上述1-out-of-2 OT中的问题基本相同，唯一变化的就是从 2 条秘密传递 1 条给 Bob 变成了 从 n 条秘密传递给 Bob .

初始协议为：n条秘密为： ,且每个消息长为 bits，设 表示第 i 个秘密的第 j 位，可见 。

> (1)首先 Alice 随机选择大素数 ，计算 ，并计算模 m 的二次非剩余 y，然后对于每一个比特位 选择一个整数 并计算 （显然，当且仅当 时 是二次剩余）然后将 发送给 Bob。  
> (2)Bob选择随机数 r 以及随机的比特位 ，计算 ，其中当且仅当 时 ，q 是 m 的二次剩余。  
> 如此一来，只需要验证 (2) 中得到的 q 是否为 m 的二次剩余即可，若是则 反之则不等。  
> 每一轮上述过程 Bob 可以得某个消息的一个比特位，重复 次，即可得到秘密 。

虽然上述方法可以完成目标，但是仍存在三个缺点：

1. Bob 可能询问的是不同秘密的比特位。

2. Bob 可能得到两个消息之间的异或。

3. Alice 可能欺骗 Bob，发送的 y 可能是一个 二次剩余，也就有可能指出 Bob 的选择。

为克服以上缺点，改进上述算法如下：

> (1) Bob 随机选择扰动函数 ，随机整数 和随机比特位 ，使得 ，其中 i 表示 Bob 想要获得第 i 条秘密。并计算 将 t 个 传送给 Alice，同时需要向 Alice 证明所有的 可用性（我的理解是以此来保证请求的是同一条消息的不同比特位，也就是解决缺点 1和2）。  
> (2) Bob 传送 k 给 Alice ( ).  
> (3) Alice 对于每一个 都给出模 m 下的二次特征值，发送给Bob。  
> (4) Bob根据二次特征值来推测出相应比特的数据（特征值为零则与 α 不同，不为零则相同）。

这便是最开始的 1-out-of-n OT 协议。至此最基本的OT协议包括 2 取 1，n 取 1 都已经有了。接下来就是关于OT扩展的内容了。

Beaver 96

但是在1988年，Impagliazzo和Rudich\[4\]就已经证明了不能使用黑盒的构造方法从一个单项函数来实现 OT，也就是说，虽然OT协议的设计已经很成熟，但是应用中的运算消耗仍然很高。而公钥密钥的出现有助于解决现有的OT协议执行次数过高的问题，不过因为公钥密钥需要低效率的幂指数运算，不利于直接加密大数据，因此将公钥的私钥两种方法结合，对现有OT进行扩展，提高OT协议的效率。1996年Beaver\[5\]等人就使用这种方法，提出了一种OT扩展技巧，通过伪随机装置来对等大量OT执行的效果。伪随机依靠一个单向函数（一个黑箱）来了实现。

下图是一个基本的1-out-of-2 OT, 设预计算阶段的 R-OT 协议使发送方 Alice 获得两个随机信息 , 接收方Bob获得 , 设在线阶段Alice的需要发送的信息为 , Bob 的选择比特为 . Beaver 去随机化的主要思想是 Alice 使用R-OT的两个随机信 作为一次一密(OTP)来加密需要他发送的信息 并将盲化结果 发给 Bob. 如果 , Alice发送的两个信息是 , Bob本地计算 即可得到选择比特 c 对应的比特秘密 ,但是无法获得 。但是如果 ，按照上述步骤我们会发现 Bob 将无法获得正确的信息，除非 Alice 交换 来加密 。如下图所示：

![图片](<https://mmbiz.qpic.cn/mmbiz_jpg/5Ig98FmhHw0mZWwR4nvNUJU1fcppKu2yG79Jl84dBqkNTrB7tMZWQnb7jC94S7Ys1ruhKbic2931dQ8tDgl6krQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1>)为解决上述 的情况，容易想到的解决方法是让 Bob 告知 Alice 是否有 ，具体如下图所示，其中 是否等于0，表示为

![图片](<../../assets/44d5041f007d30a9.png>)上述过程就是一个在基本的 1-out-of-2 OT上使用随机数来实现的过程，即 R-OT 。但是其每执行以此 OT 都需要进行里现阶段的随机数生成，这个开销非常昂贵。反之其在线阶段只需要异或运算，反而比较高效。并且以此离线阶段的R-OT以此只能产生一个OT实例，这也给其应用带来困难。而Beaver等人在1996年，使用了混合加密的方式对R-OT进行了改进，使得其离线阶段效率得到了提高，并且例证了OT扩展方案的可行性。

要讲清Beaver等人的 OT 扩展，首先了解一下混合加密。公钥密钥的出现有助于解决现有的OT协议执行次数过高的问题，不过因为公钥密钥需要低效率的幂指数运算，不利于直接加密大数据，因此将公钥的私钥两种方法结合，使用昂贵的公钥加密短密钥，然后使用相对便宜的私钥来加密长信息。以此实现对现有OT进行扩展，提高OT协议的效率。应用如下：

![图片](<../../assets/44d5041f007d30a9.png>)

如上图所示，使用姚氏混淆电路来完成整个过程，其中左侧为 Alice 输入 并得到所有的消息对，即就是 n 个R-OT离线阶段的 ，右侧为 Bob 输入 并得到比特串 r 以及对应的消息 ，即就是R-OT离线阶段的 。其中输入的 长度为 ，输出的结果个数为 n ，且 。也就是说在此扩展中只需要 个R-OT，可以得到 n 个 OT 实例，即就是使用 实现 ,同样有 。

然而由于姚氏混淆电路的原因导致其不够实用，但是他的例证说明了 OT-extension 是可行的。也为后面 OT 的发展做出重要贡献。

IKNP\[03\]

在Beaver等人之后，IKNP\[03\]\[6\]协议则逐步让 OT 协议走上实用的道路。IKNP\[03\]协议和Beaver等人解决的问题实际上是相同的，都是为了让 Alice 获得消息对 和让 Bob 后的对应的 以便在线阶段 OT 的使用。与不太实用的Beaver96使用 Yao’s GC 不同的是，IKNP\[03\] 使用 OT 来实现，具体如下：

首先，Bob 随机构造长度为 比特串 ,然后将其看为一个 的列向量，并对每个比特位进行按行扩展为 位 (重复编码扩展)，完成后进行秘密分享，如图：

![图片](<../../assets/44d5041f007d30a9.png>)接着 Alice 也随机选取长度为 的比特串 ，然后双方共同执行base-OT 其中 Bob 为发送方，将两个秘密份额矩阵 的第 列作为输入的两个消息，Alice 作为接收方，以 为输入来选择两个秘密份额矩阵第 列的某一个来构成自己的矩阵 的第 列，结果如下:

![图片](<../../assets/44d5041f007d30a9.png>)

当 时, 两个秘密份额矩阵 对应位置的行向量相等，所以Alice所得矩阵 的行向量 与 Bob 的任意一个份额矩阵该位置对应的行向量 相同；当 时, Alice所得矩阵 的行向量 等于 Bob 的份额矩阵 该位置对应的行向量 与 的异或，即：

其中 是份额矩阵 的行向量，上述过程可以进一步抽象为：

上述过程后 Alice 获得了矩阵 ，加上自己拥有的选择串 ，则 Alice 就有了消息对 。Bob 则拥有了相应的 (选择比特), (某个消息)，如下图所示：

![图片](<https://mmbiz.qpic.cn/mmbiz_png/5Ig98FmhHw24NH8EGNIyfSiblAYEPv3KKAEUWJiaAJicj0NiazSgjA4SahmyBJdAmQ4sGUiccYfLO2myouLfMicNJmjw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1>)

运用上述公式进一步改写为：

![图片](<https://mmbiz.qpic.cn/mmbiz_jpg/5Ig98FmhHw0mZWwR4nvNUJU1fcppKu2yLNrHh2CZrSeOhF4J061PPEibHQTHianTVAPrVPDuz8X5ibVPXfuoncfXQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1>)由此就已经可以明显的看出 Alice 拥有了随机的消息对，并且 Bob 拥有选择比特位和 Alice 消息对中的其中一个消息。这已是解决一开始问题（两方获得用于在线阶段的消息队等）。效率方面，扩展矩阵是 的矩阵，其中 ，而base-OT阶段是按照列来进行，所以会进行 次 base-OT，然而传输了 n-bits 的内容，并且不需要 Yao’s GC。

但是仔细观察可以看到，上述的消息对之间重复的使用了同一个比特串 ，这使得生成的消息之间存在相关性，所以必须解决这种下相关性。本协议采用的方法是：使用一个随机预言机(Random Oracle)来解决相关性，用哈希函数 来实现随机预言机(Random Oracle) 。对于消息 和比特串 使得 是独立伪随机的。从而解决上述问题！整个INKP\[03\]协议如下图所示：

![图片](<https://mmbiz.qpic.cn/mmbiz_jpg/5Ig98FmhHw0mZWwR4nvNUJU1fcppKu2yPzgQWH74bpYbnrcfpxENRSnQYdd1uShibb0ibBBzv7EIIjuYX0jNlMFA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1>)最后整合为可用的结果为：

![图片](<../../assets/44d5041f007d30a9.png>)

此协议是面对半诚实模型的。对于 Bob 恶意的情况下，该模型是危险的！

![图片](<../../assets/44d5041f007d30a9.png>)

若 Bob 在生成的某一个份额矩阵中修改 对应行向量的某一个比特位 (如 中黑色方块由原来的 0 变为 1) ，并且同时 Alice 对应的选择向量 对应的比特位 ( Alice 得到的矩阵中橘色方块位置)，那么就会造成 Alice 得到的向量与 Bob 具有的行向量不同 (灰色块所示的向量)。在后续协议执行过程中，例如哈希等，只要Bob检测到 Alice 的结果与自己的不同，就可以推断出该位 ，相同则推断出 。如此一来，Bob得到 Alice 选择串 的一个比特值。因此 IKNP\[03\] 仅仅是半诚实安全。

KK\[13\]

在 Kolesnikov，Kumaresan\[7\]等人 2013 年发表的文章中，对 IKNP\[03\] 协议在 GMW 中长度扩展步骤的通信开销远高于核心归约步骤的通信开销问题进行了优化。Kolesnikov 等人发现，IKNP\[03\] 在base-OT 阶段，对 Bob 的选择比特串 进行扩展的时，使用的是重复编码扩展方法，这是最简单的编码方法，其编码效率仅为 ， 是扩展后的向量长度。因此 Kolesnikov 等人从此着手，使用更加复杂的编码方式在优化 IKNP\[03\] 协议。

用 表示对 使用某种编码方法，用编码的视角看待 IKNP\[03\] 协议，首先是 Bob 对选择向量 扩展，扩展后的矩阵如下：

![图片](<../../assets/44d5041f007d30a9.png>)

然后再运用秘密分享得到：（其中 示 其中一个份额矩阵的行向量）

![图片](<../../assets/44d5041f007d30a9.png>)

通过秘密分享矩阵得到 Alice 的向量组 ，然后每一个向量分别异或 ，得到 ：

![图片](<https://mmbiz.qpic.cn/mmbiz_png/5Ig98FmhHw24NH8EGNIyfSiblAYEPv3KKGAf5GiasXU1hl0wWB6mzpL6mOWXgGBAWjsFdhHN9tz86vHSE0990GXg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1>)

使用 ,改写 矩阵为：

![图片](<https://mmbiz.qpic.cn/mmbiz_png/5Ig98FmhHw24NH8EGNIyfSiblAYEPv3KKJKVPic9M80S2c1BeibwoiarRef8Baz86zdlgKG31uez3HkWIaiaVEdshtQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1>)可以看到最后 Alice 获得结果进一步变为： ，然后再用随机预言机来破坏其关联性： Bob的值也就变为 。

从编码的方式了解了 IKNP\[03\] 之后就可以使用其他的编码方式对其推广，按照这种方式推广的结果相比于 IKNP\[03\] 也就是 Alice 获得的两个消息结果中的编码部分改变而已，例如用 表示一个编码结果 ( 表示编码结果的第 位)，则 Alice 获得的两个消息变为 。而 Bob 的结果并不变。所以改进后协议的提升也都来自编码效率的改变。

ALSZ\[13\]

ALSZ\[13\]\[8\]在 IKNP\[03\] 基础上多通信复杂度和计算复杂度都进行了优化，首先是算法方面的计算复杂度，Asharov 经过实验发现 IKNP\[03\] 协议中大约 42% 的计算耗费在矩阵转置上，于是对于矩阵给的转置进行优化，如下图所示：

![图片](<https://mmbiz.qpic.cn/mmbiz_jpg/5Ig98FmhHw0mZWwR4nvNUJU1fcppKu2y3nN6mosd9C2nBuYxgxib6TFAmqnreCDPZWQaxB0Mu3Cjdib2ezsQoqxw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1>)

先对最小的 2×2 子矩阵进行转置，只需要消耗很少的时间，进而再将该 2×2 矩阵看为整体再寻找下一个 “2×2” 子矩阵。并且因为转置之间不冲突，再同一个层级上可以并行。如此一来将 m×n 的矩阵转置的计算消耗从 降低为 。其中 是CPU 寄存器大小。此外因为 拥有的 组成的矩阵列之间没有相关性，所以可以在 OT 中使用并行算法加速。

在通信复杂度方面，Asharov 等人使用盲化因子 来初始化 Bob 持有的矩阵而不是随机生成，这样每次只需要向 Alice 传输一个消息就可，具体如下。

原本的协议是：(其中 PRG() 是伪随机序列生成器)

![图片](<../../assets/44d5041f007d30a9.png>)

优化后：

![图片](<../../assets/44d5041f007d30a9.png>)

在 IKNP\[03\] 中有： ，以此来验证一下上述优化的正确性，当 时，易得 ；当 时，易得 ，这符合 IKNP\[03\] 中的结论。在通信中，原本的方案需要发送消息对 ，但是优化后的方案对于一个消息对，只需要发送一个消息 就可以了，可见该优化方法理论上可以降低此过程一半通信消耗。

Asharov 等人还对应用在 Yao’S GC 和 GMW 上的 OT-Extension 做出了相应的优化。

在 Yao’S GC 上，对于最后一部分的 OT ，使用了Correlated-OT 来降低带宽，在 C-OT 中，Alice 得到的消息对 是相关的，例如 。具体如图所示：

![图片](<../../assets/44d5041f007d30a9.png>)

是一个随机比特, ,其中 是一个全局随机量， 是Bob 的选择比特，在之前叙述的通用 OT-Extension 中，Alice 接下来需要计算并发送两个消息 。但是在应用了上述C-OT 的 Yao’S GC 上只需要Alice 设定 ，计算并发送一个消息 即可。而 Bob 根据自己的 r 值来得到自己的输出，当 ,得到 ；当 ，得到 。很容易证明，不同的 r 值会让 Bob 得到 ，所以此方法可行，并且因为只需要传输一个消息，则降低了一半的带宽需求。

在 GMW 上则使用了Random-OT 来优化，如图：

![图片](<../../assets/44d5041f007d30a9.png>)

其中 Alice 获得的 都是随机的，之后 Alice 设定 , ,而后 Bob 只需要计算 ，因为 ，显然当 ,得到 ；当 ，得到 。如此一来不需要进行最后一步的通信就可以得到消息。

最后，感谢陈小军老师的帮助、建议、指导和修改！感谢董业师兄 @酸菜鱼的帮助、建议和修改，李开运师兄 @李开运的帮助和建议！

此外上述图示中 Beaver96 和 IKNP03 部分图例是根据 Mike Rosulek 报告\[12\]中的图示重新绘制得来，特此声明！

<a id="fn_Rabin"></a>

> <sup>Rabin</sup>. Rabin M O . How to Exchange Secrets by Oblivious Transfer\[J\]. Technical Memo TR-81, 1981.[↩](<#reffn_Rabin>)

\[2\] Even S . A randomized protocol for signing contracts\[J\]. ACM SIGACT News, 1983.

\[3\] Brassard G , C Crépeau, Robert J M . All-or Nothing Disclosure of Secrets. Advances in Cryptology — CRYPTO’ 86, 1986.

\[4\] Impagliazzo R , Rudich S . Limits on the provable consequences of one-way permutations (invited talk). Springer New York, 1990.

\[5\] Beaver D . Correlated Pseudorandomness and the Complexity of Private Computations\[C\]// Twenty-eighth Acm Symposium on the Theory of Computing. ACM, 1996.

\[6\] Ishai Y , Kilian J , Nissim K , et al. Extending Oblivious Transfers Efficiently\[C\]// 23rd Annual International Cryptology Conference. CiteSeer, 2003.

\[7\] Kolesnikov V , Kumaresan R . Improved OT Extension for Transferring Short Secrets\[M\]. Springer Berlin Heidelberg, 2013.

\[8\] Asharov G , Lindell Y , Schneider T , et al. More efficient oblivious transfer and extensions for faster secure computation\[C\]// Proceedings of the 2013 ACM SIGSAC conference on Computer & communications security. ACM, 2013.

\[9\] Yao, Andrew. (1986). How to generate and exchange secrets. Annual Symposium on Foundations of Computer Science (Proceedings). 10. 162 - 167. 10.1109/SFCS.1986.25.

\[10\] Goldreich, Oded & Micali, S. & Wigderson, Avi. (1987). How to play ANY mental game. 218-229. 10.1145/28395.28420.

\[11\] V. Kolesnikov and T. Schneider. Improved garbled circuit: Free XOR gates and applications. In International Colloquium on Automata, Languages and Programming (ICALP’08), volume 5126 of LNCS, pages 486–498. Springer, 2008.

\[12\] Mike Rosulek. web.engr.oregonstate.edu
