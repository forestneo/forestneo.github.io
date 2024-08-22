---
title: "LangChain使用函数调用（Function Call）"
date: 2024-06-13
draft: false
tags: ["LangChain"]
---


# 什么是函数调用

顾名思义，就是使得我们的代码流程能调用函数。在某些地方，函数也作为工具，因此也称工具调用（Tool Calling）。函数调用中，LLM的难点在于根据用户`query`去选择哪个函数，以及对应函数的参数分别是什么。

# FunctionCall的多种方式

## 方式一

```python
import json
from dotenv import load_dotenv
from langchain.schema import HumanMessage, FunctionMessage
from langchain_openai import ChatOpenAI


load_dotenv()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def weather_function(location: str):
    match location:
        case "北京":
            weather = "晴天"
        case "上海":
            weather = "多云"
        case _:
            weather = "不清楚"

    result = [{"weather": weather}]
    return json.dumps(result)


def calculate_function(a: int, b: int):
    result = [{"计算结果": a + b}]
    return json.dumps(result, ensure_ascii=False)


def langchain_function_calling(text):
    functions = [
        {
            "name": "weather",
            "description": "了解天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "输入您想要了解天气的位置。 示例：东京",
                    },
                },
                "required": ["location"],
            },
        },
        {
            "name": "calculator",
            "description": "计算求和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "加数",
                    },
                    "b": {
                        "type": "integer",
                        "description": "被加数",
                    },
                },
                "required": ["a", "b"],
            },
        },
    ]

    messages = [HumanMessage(content=text)]
    message = llm.predict_messages(messages, functions=functions)

    if message.additional_kwargs:
        function_name = message.additional_kwargs["function_call"]["name"]
        arguments = json.loads(message.additional_kwargs["function_call"]["arguments"])

        if function_name == "weather":
            function_response = weather_function(
                location=arguments.get("location"),
            )
        elif function_name == "calculator":
            function_response = calculate_function(
                a=arguments.get("a"),
                b=arguments.get("b"),
            )

        print(arguments)  # {'a': 3, 'b': 2}
        print(function_response)  # [{"计算结果": 5}]

        messages.append(FunctionMessage(name=function_name, content=function_response))

        second_response = llm.predict_messages(messages=messages, functions=functions)
        return second_response.content
    else:
        return message.content


if __name__ == "__main__":
    s = langchain_function_calling("3+2等于多少")
    print(s)
```

## 方式3二：使用Tool

```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


@tool
def calculate_function(a: int, b: int) -> int:
    """
    计算两个数的求和

    Args:
        a: 加数
        b: 被加数
    """
    return a + b


@tool
def weather_function(location: str) -> str:
    """
    获取指定城市天气

    Args:
        location: 城市名称
    """
    match location:
        case "北京":
            weather = "晴天"
        case "上海":
            weather = "多云"
        case _:
            weather = "不清楚"

    return weather


tools = [calculate_function, weather_function]
llm_with_tools = llm.bind_tools(tools)

query = "3+2等于多少，4+2呢"
messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

print(ai_msg)

for tool_call in ai_msg.tool_calls:
    print(tool_call)
    selected_tool = {
        "calculate_function": calculate_function,
        "weather_function": weather_function,
    }[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

final_msg = llm_with_tools.invoke(messages)
print(final_msg)
```

此方式使用`tool`装饰器，要求被装饰的函数必须有docstring，否则会有`ValueError: Function must have a docstring if description not provided.`此错误。docstring为对应工具提供了说明，类似于方式一中的`parameters`。

此外，值得需要的是，方式二可以再一个查询中包括多次函数调用。比如可以回答以下问题：
- "3+2等于多少，4+2呢"
- "3+2+1等于多少？"，此问题下，final_msg依然是产生一次函数调用，并不是最终结果，需要对代码进行一定程度修改

# 参考内容
- [LangChain tool_calling](https://python.langchain.com/v0.2/docs/how_to/tool_calling/)

