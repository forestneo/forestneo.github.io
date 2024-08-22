---
title: "Advanced RAG 07"
date: 2024-05-16
draft: false
tags: ["RAG"]
---

有效解析和理解非结构化文档中的表格时，设计 RAG 方案仍然是一大挑战。以图片格式存在的表格情况下，这更困难，如扫描文件中的表格。这些挑战至少包含一下几个方面：

- 扫描文档或图像文档的复杂性，如多样化的结构、包含非文本元素以及手写和打印内容的结合，为自动准确提取表格信息带来了挑战。不准确的解析可能会破坏表格结构，使用不完整的表格进行嵌入不仅可能无法捕获表格的语义信息，而且还很容易破坏 RAG 结果。
- 如何提取表格标题并有效地将它们链接到各自的表格。
- 如何设计一个索引结构来有效地存储表格的语义信息。
  本文首先介绍管理 RAG 中表格的关键技术，然后回顾一些现有的开源解决方案，再提出并实施一个可能的解决方案。

# 关键技术
## Table Parsing（表格解析）

![](https://forest-pic.oss-cn-beijing.aliyuncs.com/image-20240501152928724.png)

这个模块的主要功能是从非结构化文档或图像中准确提取表格结构。最好能够提取相应的表格标题，并方便开发者将表格标题与表格关联起来。目前有以下几种方法:

1. 使用多模态 LLM（multimodal LLM）如 GPT-4V 来识别表格并从每个 PDF 页面中提取信息
2. 利用专门的标检测模型，如[Table Transformer](https://github.com/microsoft/table-transformer "Table Transformer")来辨别表结构
3. 使用开源框架，如[unstructured](https://unstructured-io.github.io/unstructured/best_practices/table_extraction_pdf.html "unstructured")或者其他框架，这些框架允许对整个文档进行全面分析，并从分析的结果中提取与表相关的内容。
4. 使用端到端模型（如 Nougat、Donut 等）解析整个文档并提取与表格相关的内容。此方法不需要 OCR 模型。
   值得一提的是，无论使用何种方法提取表格信息，都应包含表格标题。这是因为在大多数情况下，表格标题是文档或论文作者对表格的简要描述，可以在很大程度上总结整个表格。在以上几种方法中，第四种可以相对较容易地对表格标题进行检测以及和表格进行关联。

## Index Structure

索引表格的方式大概可以分为以下几种

1. 仅对图像格式的表格进行索引
2. 仅对纯文本或 JSON 格式的表进行索引
3. 仅索引 LaTeX 格式的表格
4. 仅索引表的摘要（表的摘要通常可以使用 LLM 进行生成，也可以尝试使用多模态模型）
5. 从小到大或按照文档摘要索引结构（小块内容可以是表中的行，大块内容是图像、文本或者 LaTeX 格式表格），如下图所示
![](https://forest-pic.oss-cn-beijing.aliyuncs.com/202405082223921.png)

也有一些算法不需要对表进行解析，比如可以将相关的 PDF 页面（图像）和用户查询发送到 VQA 模型（如 DAN）或者多模态 LLM（如 GPT-4V），即可返回答案，如下图所示：

![](https://forest-pic.oss-cn-beijing.aliyuncs.com/202405082229654.png)

同时，也可以将图片，文本块，用户查询发送到多模态 LLM 直接得到答案。

值得注意的是，_并不是所有的方法都依赖于 RAG 流程_，比如：

- 在特定数据集上进行训练，并使模型（类似 BERT 的 transformers）能够支持表理解的任务，如[TAPAS](https://aclanthology.org/2020.acl-main.398.pdf "TAPAS")方法。
- 使用 LLM 采用预训练、微调或者提示词的方式，是的 LLM 可以执行表格理解任务，如  [GPT4Table](https://arxiv.org/pdf/2305.13062.pdf "GPT4Table")方法。

# 现有开源解决方案

LlamaIndex 中提供了四种方法，其中前三种依赖多模态模型：

1. 检索相关图像，并将其发送到 GPT-4V 进行回复
2. 将每个 PDF 页面作为图像，让 GPT-4V 对每个页面进行 reasonings，并对其建立文本索引，然后根据 Image Reasoning Vector Store 进行查询答案
3. 使用 Table Transformer 从检索到的图像中裁切表信息，然后使用 GPT-4V 进行回答
4. 对裁剪后的表格图像 OCR 处理，将文本发送到 LLM 中进行回答
   Langchain 中也有一些解决方案，有以下几类：

- [半结构化 RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb?ref=blog.langchain.dev "半结构化RAG")：首先使用 Unstructured 从 pdf 中解析文本和表格，然后使用 multi-vector retriever 存储原始的表格和文本，同时对表格进行总结，最后用 LCEL 实现问答链路。主要流程如下所示（实际上这里的最后一个 LLM 并不是多模态 LLM）
 ![](https://forest-pic.oss-cn-beijing.aliyuncs.com/202405082245122.png)
- [半结构化与多模态 RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb "半结构化与多模态RAG")：这个当中实际上提出了多种方案，如下图所示。 - _Option1_：和半结构化 RAG 类似，使用多模态 embedding（如 CLIP）对图像和文本进行 embedding，然后将原始图像和文本块传入到多模态 LLM 进行回答 - _Option2_：也是使用多模态 LLM，根据图像生成文本摘要，然后对文本进行 embedding，这样通过传统 RAG 链路即可进行 QA - _Option3_：从图像生成文本摘要，然后进行 embedding，但是 QA 的时候，根据 embedding 找到原始的图像，并使用多模态 LLM 进行回答
![](https://forest-pic.oss-cn-beijing.aliyuncs.com/202405082246685.png)

# 建议的方案

本文对关键技术和现有解决方案进行了总结、分类和讨论。基于此，我们提出了以下解决方案，如下图所示。为简单起见，这里省略了一些 RAG 模块，例如重新排名和查询重写。
![](https://forest-pic.oss-cn-beijing.aliyuncs.com/202405082255804.png)

- 表格解析：采用 Nougat，根据作者的描述，他的表检测功能比 unstructured 更强大，同时也能很好地提取表格标题，非常方便与表格进行关联
- 文档摘要索引：小块的内容包括表格摘要，大块的内容包括 LaTeX 格式的表格和标题，然后采用 Multi-Vector Retriever 策略存储
- 表格摘要：将表格和表格标题发送给 LLM 进行汇总

这里介绍一下[Nougat](https://arxiv.org/pdf/2308.13418.pdf "Nougat")，它基于[Donut](https://arxiv.org/pdf/2111.15664.pdf "Donut")开发，不需要与 OCR 相关的模块，总体架构图为：
![](https://forest-pic.oss-cn-beijing.aliyuncs.com/202405082258961.png)
Nougat 解析公式表格的能力令人印象深刻，也能对表格标题进行关联（解析出来的表格式 LaTeX 格式），比如下面这个例子：
![](https://forest-pic.oss-cn-beijing.aliyuncs.com/202405082259145.png)

采用 Nougat，有以下有缺点：

- Pros（优点） - 可以将公示和表格等比较难解析的部分解析为 LaTeX 源码 - 解析的结果是类似于 Markdown 的半结构化文档 - 可以轻松获取表格标题并与表关联
- Cons（缺点） - 解析速度慢，大规模使用可能受限 - Nougat 是基于科研论文进行训练的，因此对于类似格式比较友好，对于其他格式可能效果一般 - 双栏中解析表格不如单栏中效果好

# 总结

本文讨论了 RAG 流程中表处理的关键技术和现有解决方案，并提出了解决方案及其实现。本文推荐使用 nougat 来解析表格。但是，如果有更快、更有效的解析工具可用，我们也建议对其进行替换。对工具的态度是先有正确的想法，然后找到工具来实现它，而不是依赖某个具体的工具。在本文中，我们将所有表内容输入到 LLM。但是，在实际场景中，我们应该考虑表超过 LLM 上下文长度的情况，比如通过有效地分块方法来在一定程度上解决这个问题。
