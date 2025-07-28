---
title: "强大的包管理工具uv"
date: 2025-07-07
draft: false
tags: ["Python"]
---

之前在项目开发过程中，使用 poetry 来管理 python 项目，一度认为其就是最好用的。poetry 的特点：
- 使用 `pyproject.toml` 和 `poetry.lock` 管理依赖，和 pip 中的 requirements.txt 相比，显然更简洁。
- 不具备管理 python 解释器的功能，需要和 pyenv 搭配使用
- 无法设置全局的镜像源，可设置单个项目镜像源
- 安装包感觉有时候比较慢
- 提供从开发到发布的完整流程

这几年，[uv](https://docs.astral.sh/uv/) 的出现极大撼动了 poetry 的江湖地位。uv 的官网介绍是这么说的：An extremely fast Python package and project manager, written in Rust，uv 有以下特点：
- 比 pip 快 10-100 倍
- 兼容 `pyproject.toml`
- 一个工具，搞定 python 环境管理，项目环境隔离。pyenv 也不需要了。

# 安装
有很多种方式可以安装，安装了 brew 的话，使用 brew 是个不错的选择。更多安装方式可参考：《[uv-installation](https://docs.astral.sh/uv/getting-started/installation/)》。

```bash
# 使用brew安装
brew install uv

# 使用curl安装
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# uv 常见使用命令

## python 环境管理

```bash
# 安装python版本
uv python install 3.11 3.12

# 查看已安装/未安装的python环境
uv python list

# 卸载
uv python uninstall 
```

## 项目管理
```bash
# 创建项目
uv init project_name
uv init project_name --python 3.11

# 在项目内初始化项目
uv init
# 安装依赖
uv sync

# 创建虚拟环境
uv venv
uv venv --python 3.12.0

# 初始化虚拟环境，交给vscode即可
source .venv/bin/activate

# 安装python包
uv add pandas langchain

# 查看包
uv pip list
```

