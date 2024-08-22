---
title: "使用Hugo构建个人主页"
date: 2024-07-11
draft: false
---

碰到的问题：运行`hugo server`之后，在`public/intex.html`文件中，stylesheet的相关链接依然是localhost。这将导致部署到了github pages中之后，网站找不到对应的css文件，因此显示效果不如人意。如下所示：

![](https://forest-pic.oss-cn-beijing.aliyuncs.com/202407112103230.png)

本质问题应该是`hugo.toml`中`baseURL = 'https://forestneo.github.io'`未生效。这个问题也在官网上发现了同样的，但是没找到解决方法。

为了暂时解决这个问题，我暂时的方法是本地预览成功之后，在代码库中进行一个全局替换，把`http://localhost`地址替换成了`https://forestneo.github.io`。未来看到别的好办法再来描述。
