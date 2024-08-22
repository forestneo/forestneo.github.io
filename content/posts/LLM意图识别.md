---
title: "LLM/RAG如何进行意图识别（路由）"
date: 2024-06-14
draft: false
tags: ["Agent", "RAG"]
---


# 为什么需要进行意图识别

![](https://forest-pic.oss-cn-beijing.aliyuncs.com/202406172049133.png)

RAG 的召回、生成过程逻辑是不够灵活的，无法解决用户的所有问题。在实际过程中，RAG 更适合作为一个子流程。即业务中往往需要首先识别用户 query 中的意图，再进入不同的分支流程进行响应（如图：[semantic routing](https://www.linkedin.com/pulse/smarter-chatbots-how-semantic-routing-prevents-unwanted-fuwte "semantic routing")）。这样总的 RAG 应用程序就是更具鲁棒性了。

意图识别也称作路由，可能作用于以下场景：

1. 路由，如可能进行`RAG`，也可能进行`Text2SQL`，也可能进行工具调用
2. 用户的`query`可能有不同的分析路径，如`Agent`、向量检索或直接交给`LLM`进行分析
3. 即使是同一个分析模式，可能需要根据`query`动态地选择数据库
4. 不同的`query`可能对应不同的`prompt`模版，路由可以帮忙选择`prompt`，比如选择小红书风格或知乎风格的写作。

当然，并不是所有的场景一定需要`router`，我们也可以通过产品逻辑引导用户，比如对于不同的`prompt`，可以让用户在请求前通过交互进行选择（难免有些显得不够智能了^\_^!）。

要想做得更好，我们面对的应该就是用户的原始自然语言请求，然后**我们根据用户的自然语言输入做出相应的分支处理**。

# Router 的分类实现

目前，可以考虑以下几种意图识别方法：

- 基于逻辑路由
- 通过 LLM prompt
- 语义 router：semantic-router

值得注意的是，这几个概念，比如逻辑路由，prompt 路由等，都是我自己编的~

## 基于逻辑/关键字路由

这部分路由最容易理解，类似于我们平时写`if/else`语句，只不过这里可以是任意的自定义规则，如关键字、正则匹配、字符串长度等。

`Haystack`中，`ConditionalRouter`的示例就是根据`query`的长度返回使用哪个分支。参考链接为：https://docs.haystack.deepset.ai/docs/conditionalrouter。

关键字路由可以认为是另一类逻辑路由，通过匹配关键字与路由分支来决定子流程。当然，提取关键字的策略是多样的，也可以利用`LLM`进行关键字提取，那样的话就可以认为不是基于逻辑的路由了。

## LLM prompt Router

顾名思义，`prompt router`通过`prompt`引导`LLM`输出对应的分支（我就是这么取名字的）。在 LangChain 的[routing 示例](https://python.langchain.com/v0.1/docs/expression_language/how_to/routing/ "routing示例")中，用了这样的`prompt`：

```python
PromptTemplate.from_template("""Given the user question below, classify it as either being about `LangChain`, `Anthropic`, or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
```

对于用户`query`，让`LLM`输出是属于哪个类别，这样后面再接一个`RunnableLambda`或者`RunnableBranch`去执行不同类别的子流程。这里需要注意的是，虽然`prompt`中规定了让`LLM`输出指定的三个类别，`LLM`是可能不听话的。因此在后面的流程判断中，需要做兜底处理。

此外，这也是一种利用LLM实现router的方式，参考[Adaptive RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/)，相关代码为：

```python
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
print(
    question_router.invoke(
        {"question": "Who will the Bears draft first in the NFL draft?"}
    )
)
print(question_router.invoke({"question": "What are the types of agent memory?"}))

# datasource='web_search'
# datasource='vectorstore'
```

此示例代码看上去是使用了`prompt`，实际上也结合了`LLM`的`Function Calling`功能。



## 函数调用 Router

`LLM`的`Function Calling`功能允许`LLM`根据用户`query`解析出要访问的函数和参数。通过将不同的分支描述成不同的函数或者工具，即可让`LLM`进入到子分支实现路由的功能。Langchain中的介绍页面为：[Function Calling](https://python.langchain.com/v0.1/docs/modules/model_io/chat/function_calling/)，以下是部分示例：

```python
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class add(BaseModel):
    """Add two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


tools = [add, multiply]
always_call_tool_llm = llm.bind_tools([add, multiply])

query = "What is 3 * 12? Also, what is 11 + 49?"

print(llm_with_tools.invoke(query).tool_calls)


```

输出信息为：

```python
[{'name': 'multiply',
  'args': {'a': 3, 'b': 12},
  'id': 'call_UL7E2232GfDHIQGOM4gJfEDD'},
 {'name': 'add',
  'args': {'a': 11, 'b': 49},
  'id': 'call_VKw8t5tpAuzvbHgdAXe9mjUx'}]
```

官网的示例中，`bind_tools`方法可以指定`tool_choice`参数控制是否一定调用工具，但是我实现的时候，报错了，没有继续去深究。

## Semantic-Router

这是一个开源项目，地址为：https://github.com/aurelio-labs/semantic-router，其原理是为每个分支提供一系列`query`示例，然后选择最相似的`query`，返回对应的分支。

```python
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import CohereEncoder, OpenAIEncoder

politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)

chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

encoder = CohereEncoder()
rl = RouteLayer(encoder=encoder, routes=[politics, chitchat])
```

如上述代码创建了两个分支，每个分支下有一些`query`示例。其原理为对于用户请求，从所有的

# 零样本分类

Zero-Shot Classification 是 NLP 中的一类任务，其在一组标记的示例上进行训练，然后可以对未知的类进行分类，可以认为是一种迁移学习。下述代码展示了使用`bart-large-nmli`进行分类：

```python
from transformers import pipeline

pipe = pipeline(model="facebook/bart-large-mnli")
pipe("I have a problem with my iphone that needs to be resolved asap!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)

# output
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}
```

Heystack 中的[ZeroShotTextRouter](https://github.com/deepset-ai/haystack/blob/main/haystack/components/routers/zero_shot_text_router.py#L25 "ZeroShotTextRouter")就使用了`MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33`模型。

# 总结

本文讨论了意图识别在增强 RAG 应用方面的重要性。通过意图识别，用户可以根据用户的自然语言输入自动选择不同的处理路径，如 RAG、Text2SQL 或工具调用。同时介绍了几种实现方法，包括基于逻辑规则、通过 LLM 提示、函数调用、语义路由和零样本分类登，帮助系统更智能地响应用户查询。

# 参考内容

- [Routing in RAG-Deiven Applications](https://towardsdatascience.com/routing-in-rag-driven-applications-a685460a7220 "Routing in RAG-Deiven Applications")
- [semantic-router](https://github.com/aurelio-labs/semantic-router "semantic-router")



