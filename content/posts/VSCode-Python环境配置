---
title: "VSCode配置Python环境"
date: 2025-12-25
draft: false
tags: ["Coding"]
---

我创建了一个 python 项目，大郅的路径如下
```
project_root/
  ├── app/
  │    ├── utils
  │    └── server
  ├── .vscode/
  └── ...
```

在 server 中的 python 文件中，我希望直接 `from utils.xx import xx`，但是这时候会报错，因为 app 这个路径不在 python 的环境变量里。

这个时候比较优雅的方式是将我这个项目变成一个开发中的包。但是由于需要代码部署并上线，远程环境和我本地环境不一致，因此在我这个项目中，将相关的 package 放在 app 是必要的。

按照网上的说法我使用了 vscode 的 `launch.json` 配置，但是我发现不起作用，设置如下：

![](https://forest-pic.oss-cn-beijing.aliyuncs.com/202512191052227.png)

这主要是因为：`launch.json` 只在你使用调试模式时生效。

- ❌ 如果点击的是右上角的 `Run Python File`： 那个按钮不读取 `launch.json`。它读取的是 `settings.json` 或者终端的默认配置。
- ✅ 如果你按的是调试键： 那么 `launch.json` 是应该生效的。

解决方法：在项目根目录的 `.vscode/settings.json` 文件中，添加以下内容：

```json
{
    "terminal.integrated.env.osx": {
        "PYTHONPATH": "${workspaceFolder}/app"
    },
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "${workspaceFolder}/app"
    },
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${workspaceFolder}/app"
    }
}
```

这时候，通过快捷键或者 Run Python File 运行 python 脚本时，就能找到对应的 python 路径了。
