---
title: "Python-一次调试装饰器中的bug记录"
date: "2022-09-11T15:27:53+08:00"
updated: "2022-09-11T15:27:53+08:00"
categories: ["编程语言", "Python"]
tags: ["Python"]
author: "DPer"
original_url: "http://forestneo.top/2022/09/11/编程-Python-一次调试装饰器中的bug记录/"
source_html: "ForestNeo-website-master/2022/09/11/编程-Python-一次调试装饰器中的bug记录/index.html"
---

本文档记录了实现切面当中很长时间发现的一个潜在bug，其实和代码本身无关，而是调试过程中IDE有关。如果没有切面和`python`装饰器的基础，可以直接看最后一个章节当中bug的复现。

<a id="基础功能实现"></a>

# 基础功能实现

首先，由于业务需求，我们封装了一个自己的数据结构，为了方便，就以`spark`当中的`DataFrame`封装为例。首先，我们读取一个简单的表格：

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.options(header='True', inferScheme='True').csv("test.csv")
df.show()
```

假设这个数据表长这样：

```
+---+---+
|  a|  b|
+---+---+
|  1|  2|
|  2| 30|
|  3| 14|
+---+---+
```

首先，由于业务需求，我们需要封装一个自己的数据结构，就叫他 `MyDataFrame` 吧，大致框架如下：

```python
from pyspark.sql import DataFrame as SparkDataFrame

class MyDataFrame:
    def __init__(self, sdf: SparkDataFrame):
        self.sdf = sdf

    def __getattr__(self, item):
        print("in __getattr__, item = ", item)
        # code here
        return self.sdf[item]

    def __getitem__(self, item):
        print("in __getitem__, item = ", item)
        # code here
        return self.sdf[item]
```

在上面的代码中，`__getattr__`和`__getitem__`是因为用户可能会通过dataframe获取某一列，即为了满足`df.a`和`df['a']`这两种用法。因此，到目前为止，就支持以下用法了：

```python
if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.options(header='True', inferScheme='True').csv("write_table.csv")
    df.show()
    mdf = MyDataFrame(sdf=df)
    # __getattr__(self, 'a')    
    col1 = mdf.a
    # __getitem__(self, 'a')
    col2 = mdf['a']
```

运行一下，也没有问题，结果为：

```
+---+---+
|  a|  b|
+---+---+
|  1|  2|
|  2| 30|
|  3| 14|
+---+---+

in __getattr__, item =  a
in __getitem__, item =  a
```

<a id="日志功能添加"></a>

# 日志功能添加

在实现了这个功能之后，产品突然提需求了。说希望给自己的每个 API 都加上日志，查看运行的时间以方便后续进行性能优化。毕竟咱们写代码的是面向产品进行服务，那他说的也合理，就得开干了。但是作为写代码的，我们也得好好思考自己怎么实现则个需求，核心目标就以下几点：

- 实现产品要求的功能

- 效率要高，要是成千行代码要加这个功能，这得加到哪天，以及哪天加了其他的api，难以保持log功能一致

- 还是易维护，哪天产品不要这个功能了，得好剥离掉

于是就想到了通过装饰器来实现这个功能。首先，需要实现这样一个装饰器：

```python
import time
from functools import wraps

def log(func):
    @wraps(func)
    def my_loging(*args, **kwargs):
        op = func.__name__
        print("in log: op = ", op, args)

        # 额外的事情
        a = 1

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("LOGGING:: func={}, time_used={:.6f}s\\n".format(op, end_time-start_time))

        return result
    return my_loging
```

这样的话，核心的`MyDataFrame`这么写就可以了，精准且优雅，😏~

```python
class MyDataFrame:
    def __init__(self, sdf: SparkDataFrame):
        self.sdf = sdf

    @log
    def __getattr__(self, item):
        print("in __getattr__, item = ", item)
        # code here
        return self.sdf[item]

    @log
    def __getitem__(self, item):
        print("in __getitem__, item = ", item)
        # code here
        return self.sdf[item]
```

然后我们重新运行一下原有的代码，结果是这么个样子：

```
in log: op =  __getattr__ (<__main__.MyDataFrame object at 0x7fbfaed66320>, 'a')
in __getattr__, item =  a
LOGGING:: func=__getattr__, time_used=0.000691s

in log: op =  __getitem__ (<__main__.MyDataFrame object at 0x7fbfaed66320>, 'a')
in __getitem__, item =  a
LOGGING:: func=__getitem__, time_used=0.000326s
```

<a id="Bug-产生了"></a>

# Bug 产生了

在`log(func)`的代码中，有的时候并不只是记录时间，还会去干别的事情，这个时候还可能会去对代码进行调试。为了让代码更健壮，我们对可能得异常进行捕获，那现在代码就变成了这个样子了：

```python
def log(func):
    @wraps(func)
    def my_loging(*args, **kwargs):
        op = func.__name__
        print("in log: op = ", op, args)

        try:
            # 额外的事情
            a = 1

            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print("LOGGING:: func={}, time_used={:.6f}s\\n".format(op, end_time-start_time))
            return result

        except Exception as e:
            print("error")
            print(e)

    return my_loging
```

这个时候，代码是可以正常运行的，但是，如果开始调试，就会出现bug。假如我在`a=1`那里添加了一个断点，然后开始运行：

![](<https://forest-pic.oss-cn-beijing.aliyuncs.com/20220909184235.png>)

在监控窗口中，如果把`args`展开，就会报错，报错的信息截图如下：

![](<https://forest-pic.oss-cn-beijing.aliyuncs.com/20220909184151-20220909184214779.png>)

这个错误只会在展开`args`的瞬间出发，并且只在调试过程中产生。甚是神奇。

<a id="Bug分析与防范"></a>

# Bug分析与防范

在上面的演示过程中，实际上，目前我碰到了两类可导致这个问题的错误：

- `MyDataFrame.shape`

- `len(MyDataFrame)`

目前个人的猜测为：在调试的过程中，调试的窗口会对监控的对象执行 len(Object) 和 Object.shape 去获取其信息，这个时候就会触发代码里面的对应操作。而我们在代码中，没有去实现 `__getattr__(shape)`这个操作，这最终导致了 bug 的产生。对于`len(Obj)`也同理。

<a id="Bug-复现"></a>

## Bug 复现

这个问题的本质其实和装饰器没关系，我复现了一下，只需要一下基本的代码，就可以在调试过程中产生意外的信息：

```python
class MyDataFrame:
    def __init__(self, sdf: SparkDataFrame):
        self.sdf = sdf

    def __getattr__(self, item):
        print("in __getattr__, item = ", item)
        # code here
        return self.sdf[item]

    def __getitem__(self, item):
        print("in __getitem__, item = ", item)
        # code here
        return self.sdf[item]

if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.options(header='True', inferScheme='True').csv("write_table.csv")
    df.show()
    mdf = MyDataFrame(sdf=df)
    col1 = mdf.a
    col2 = mdf['a']
```

在`col1=mdf.a`这一行代码添加断点，即可在 console 看到系统调用了`mdf.shape`了。如下图所示：

![](<https://forest-pic.oss-cn-beijing.aliyuncs.com/20220909184331.png>)

<a id="解决方法"></a>

## 解决方法

因此，要想此类问题不产生，我们需要对获取shape和len这两个信息进行额外的处理。在装饰器中可添加以下代码：

```python
if op == '__getattr__' and args[1] == 'shape':
    return 0
```

在自己实现的类中，建议添加`__len__()`方法。

```python
class MyDataFrame:
    def __init__(self, sdf: SparkDataFrame):
        self.sdf = sdf

    def __len__(self):
        return 0
```

当然，最好对`.shape`操作也单独处理一下。不过这当中会引发一个新的问题，如果dataframe当中，有一列的列名就叫shape，那就导致歧义了。
