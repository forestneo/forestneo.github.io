---
title: "Langchain-LCEL用法介绍"
date: 2024-04-17
draft: false
tags: ["LangChain"]
---

LCEL是LangChain引入的一种全新语法，LCEL全称为LangChain Expression Language，可以更方便地构建LLM的Pipeline。在LangChain中，LCEL支持异步、并行、流式等特点。

# LCEL 初探



举个例子，代码如下：

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI(model="gpt-4")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "ice cream"})
```

chain 的构建只是搭积木一样把 prompt，model，output_parser 拼了起来，相对于自己实现一个类，易用性更高，代码也更为简洁。

为了实现此效果，Langchain 抽象除了一个 Runnable 类，每个类中定义了以下两类方法：

- 同步方法 - `stream` - `invoke` - `batch`
- 异步方法 - `astream` - `ainvoke` - `abatch` - `astream_log` - `astream_events`
  同时，对于核心组件，其输入和输出分别为：

| 组件           | 输入类型                                              | 输出类型              |
| -------------- | ----------------------------------------------------- | --------------------- |
| `Prompt`       | dict                                                  | PromptValue           |
| `ChatModel`    | Single string, list of chat messages or a PromptValue | ChatMessage           |
| `LLM`          | Single string, list of chat messages or a PromptValue | String                |
| `OutputParser` | The output of an LLM or ChatModel                     | Depends on the parser |
| `Retriever`    | str                                                   | List of Documents     |
| `Tool`         | str or dict                                           | Depends on the tool   |

基于此，不同的组件就可以搭积木一样串起来。然后天然支持非流式和流式接口。

# 如何用 LCEL 实现业务逻辑

上一章节中，通过 `chain = prompt | model | output_parser`，可以快速搭建应用业务逻辑。但是如果仅能支持这种顺序执行的前后结果传递的逻辑，难免有点局限。比如如果想把输入并行经过两个 prompt 然后结果拼接，看上去就不太好实现了。

当然，这点设计人员自然也想到了。LCEL 可以通过不同的 Runnable 组件来搭建特定的业务逻辑。即 LCEL 有自己的"语法"。

## Runnable 类型

当前，Runnable 有以下几种类型：

- RunnableParallel
- RunnablePassthrough
- RunnableLambda
- RunnableBranch

## RunnableParallel

RunnableParallel 可以用于操作一个 Runnable 的输出，使其与序列中的下一个 Runnable 的输入格式相匹配。可以参考这个例子：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
poem_chain = ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | model

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

map_chain.invoke({"topic": "bear"})
```

最后返回的数据为：

```python
{
    'joke': AIMessage(content="Why don't bears wear shoes?\n\nBecause they have bear feet!"),
    'poem': AIMessage(content="In the wild's embrace, bear roams free,\nStrength and grace, a majestic decree.")
}
```

如果输入的 dict 中有多个键值，在流程中想取出特定键的时候，可以用 python 自带的 itemgetter。后续例子中再进行介绍。

## RunnablePassthrough

RunnablePassthrough 允许原样传递输入，或者在输入中添加额外的键。通常与 RunnableParallel 结合使用，以便将数据分配给映射中的新键。

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

runnable.invoke({"num": 1})
```

输出为：

```python
{'passed': {'num': 1}, 'extra': {'num': 1, 'mult': 3}, 'modified': 2}
```

上述案例中，涉及到 RunnablePassthrough 的两个语法：

- `RunnablePassthrough()`: 直接将自身进行传递下去
- `RunnablePassthrough.assign()`:添加新的键值

# RunnableLambda

RunnableLambda 允许在流程中使用任意自定义函数。以下代码结合了 itemgetter，RunnableParallel，RunnablePassthrough，RunnableLambda，助于理解前面提到的这些语法。

```python
from operator import itemgetter

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

chain = RunnableParallel(
    a = itemgetter("foo") | RunnableLambda(lambda x: len(x)),
    b = {"text1": itemgetter("foo"), "text2": itemgetter("bar")}
        | RunnableLambda(lambda x: len(x["text1"]) * len(x["text2"])),
    c = RunnablePassthrough()
)

r = chain.invoke({"foo": "bar", "bar": "gah"})
print(type(r))      # <class 'dict'>
print(r)            # {'a': 3, 'b': 9, 'c': {'foo': 'bar', 'bar': 'gah'}}
```

# RunnableBranch

前面提到的 LCEL 语法都是确定性的，对于前一个输入都会去执行下一个组件得到一个输出。这时候会碰到一个问题，如果下一个组件的执行逻辑依赖于输入，那么就有点棘手了。此功能即路由功能，根据数据动态决定下一步调用组件。有两种思路助于解决这个问题：

1. 利用 RunnableBranch
2. 实现自定义函数，在函数内部实现路由

首先来看 RunnableBranch，其入参为一系列 `(condition, runnable)` 对和一个 runnable，运行满足第一个 condition 对应的 runnable 并返回，如果没有找到满足的条件，则运行默认的 runnable。

```python

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch

prompt1 = PromptTemplate.from_template("""This is prompt1: {question}""")
prompt2 = PromptTemplate.from_template("""This is prompt2: {question}""")
prompt3 = PromptTemplate.from_template("""this is default prompt: {question}""")

branch = RunnableBranch(
    (lambda x: len(x["question"]) > 8, prompt1),
    (lambda x:  len(x["question"]) > 4, prompt2),
    prompt3,
)

print(branch.invoke({"question": 'hello world'}))  # 用 prompt1
print(branch.invoke({"question": 'hello'}))        # 用 prompt2
print(branch.invoke({"question": 'yes'}))          # 用 prompt3
```

当然，我们也可以把判断逻辑写在一个函数内部，用 RunnableLambda 的方式实现，比如：

```python
from langchain_core.runnables import RunnableLambda

def route_function(x: dict):
    if len(x["question"]) > 8:
        prompt = PromptTemplate.from_template("""This is prompt1: {question}""")
    elif len(x["question"]) > 4:
        prompt = PromptTemplate.from_template("""This is prompt2: {question}""")
    else:
        prompt = PromptTemplate.from_template("""this is default prompt: {question}""")
    return prompt.invoke(x)


chain = RunnableLambda(route_function)
print(chain.invoke({"question": 'hello world'}))
print(chain.invoke({"question": 'hello'}))
print(chain.invoke({"question": 'yes'}))
```

此方法将分支逻辑包在了函数内部，也可达到同样的效果。
