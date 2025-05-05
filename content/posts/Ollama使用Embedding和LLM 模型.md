---
title: "Ollama 使用 Embedding 和 LLM 模型"
date: 2024-08-26
draft: false
tags: []
---
# 安装
Ollama 是一款专为构建和管理 LLM 的应用工具。它具备一个简洁且用户友好的命令行界面，通过这一界面，用户可以轻松地部署和管理各类开源的 LLM。

![202408261752452.png|387](https://forest-pic.oss-cn-beijing.aliyuncs.com/202408261752452.png)

这个页面列出了支持 Ollama 平台的各个大型语言模型： https://ollama.com/library ，可以查看模型的更新时间、下载次数等信息。

根据官网的 [faq 文档](https://github.com/ollama/ollama/blob/main/docs/faq.md)，默认的模型存储地址为 `~/.ollama/models`，可以通过环境变量 `OLLAMA_MODELS` 对路径进行设置（改完之后记得重启 terminal 以及重启 ollama.app）。

在使用的过程中，ollama 通过终端进行交互，可运行 `ollama help` 进行了解。
# 使用 embedding 模型
以 `bge-large-zh-v1.5` 模型为例，官网地址为：[quentinz/bge-large-zh-v1.5](https://ollama.com/quentinz/bge-large-zh-v1.5)。

打开终端输入：
```
ollama pull quentinz/bge-large-zh-v1.5
```
然后就开始下载过程了，下载完之后可以看到对应的模型已经启动了。
```bash
(base) ➜  ~ ollama pull quentinz/bge-large-zh-v1.5
pulling manifest
pulling a9d3d622b517... 100% ▕████████████████████████████████████▏ 651 MB
pulling b507b9c2f6ca... 100% ▕████████████████████████████████████▏   13 B
pulling b837481ff855... 100% ▕████████████████████████████████████▏   16 B
pulling a5442ccf88b4... 100% ▕████████████████████████████████████▏  411 B
verifying sha256 digest
writing manifest
removing any unused layers
success

(base) ➜  ~ ollama ps
NAME                             	ID          	SIZE  	PROCESSOR	UNTIL
quentinz/bge-large-zh-v1.5:latest	bc8ca0995fcd	1.3 GB	100% GPU 	4 minutes from now
```

然后我们就可以访问对应的 embedding 模型，以下是一个例子

```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "quentinz/bge-large-zh-v1.5:latest",
  "prompt": "Llamas are members of the camelid family"
}'
```

当然，也有 python 包可供使用。

```python
import ollama

embedding = ollama.embeddings(
    model="quentinz/bge-large-zh-v1.5:latest",
    prompt="Llamas are members of the camelid family",
)
```

更多接口可查看此页面： https://ollama.com/blog/embedding-models
# 使用 LLM
```
ollama pull qwen3:0.6b

```
然后即可通过 http 发送请求，示例如下：
```bash
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3:0.6b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello!/no_think"
            }
        ]
    }'
```

此外，也可以在终端中这么对话：
```bash
ollama run qwen3:0.6b
```

Python 上，openai、langchain 等众多开源工具均接入了 ollama 的模型，以 openai 为例：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)

response = client.chat.completions.create(
    model="qwen3:0.6b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The LA Dodgers won in 2020."},
        {"role": "user", "content": "Where was it played?/no_think"},
    ],
)
print(response.choices[0].message.content)
```
