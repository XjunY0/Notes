### 数据类型
#### series

[_Series_](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) 是 Pandas 中最基本的一维数组形式。其可以储存整数、浮点数、字符串等类型的数据。Series 基本结构如下：

```python
pandas.Series(data=None, index=None)
```

其中，`data` 可以是字典，或者NumPy 里的 ndarray 对象等。`index` 是数据索引，索引是 Pandas 数据结构中的一大特性，它主要的功能是帮助我们更快速地定位数据。

#### DataFrame

DataFrame 是 Pandas 中最为常见、最重要且使用频率最高的数据结构。DataFrame 和平常的电子表格或 SQL 表结构相似。你可以把 DataFrame 看成是 Series 的扩展类型，它仿佛是由多个 Series 拼合而成。它和 Series 的直观区别在于，数据不但具有行索引，且具有列索引。
![DataFrame示意图](/imgs/2024-05-07/fciX7Tk6GqSC2mY5.png)
DataFrame 基本结构如下：

```python
pandas.DataFrame(data=None, index=None, columns=None)
```

区别于 Series，其增加了 `columns` 列索引。DataFrame 可以由以下多个类型的数据构建：

-   一维数组、列表、字典或者 Series 字典。
-   二维或者结构化的 `numpy.ndarray`。
-   一个 Series 或者另一个 DataFrame。

**可以有四种种方式创建一个DataFrame：**

 1. 使用一个由 Series 组成的字典来构建
```python
df = pd.DataFrame({'one': pd.Series([1, 2, 3]),
                   'two': pd.Series([4, 5, 6])})
df
```
 2. 直接通过一个列表构成的字典来生成 DataFrame
~~~python
df = pd.DataFrame({'one': [1, 2, 3], 'two': [4, 5, 6]})
df
~~~
 4. 由带字典的列表生成 DataFrame
~~~python
df = pd.DataFrame({'one': [1, 2, 3], 'two': [4, 5, 6]}) 
df
~~~
 6. 可以基于二维数值来构建一个 DataFrame
~~~python
df = pd.DataFrame([{'one': 1, 'two': 4}, {'one': 2, 'two': 5}, {'one': 3, 'two': 6}]) 
df
~~~

Series 实际上可以被初略看出是只有 1 列数据的 DataFrame。当然，这个说法不严谨，二者的核心区别仍然是 Series 没有列索引。你可以观察如下所示由 NumPy 一维随机数组生成的 Series 和 DataFrame。





### 数据读取
读取数据 CSV 文件的方法是 `pandas.read_csv()`，你可以直接传入一个相对路径，或者是网络 URL。

```python
df = pd.read_csv("https://labfile.oss.aliyuncs.com/courses/906/los_census.csv")
df
```

由于 CSV 存储时是一个二维的表格，那么 Pandas 会自动将其读取为 DataFrame 类型。

`pd.read_` 前缀开始的方法还可以读取各式各样的数据文件，且支持连接数据库。这里，我们不再依次赘述，你可以阅读 [_官方文档相应章节_](https://pandas.pydata.org/pandas-docs/stable/reference/io.html) 熟悉这些方法以及搞清楚这些方法包含的参数。


#### 基本操作

通过上面的内容，我们已经知道一个 DataFrame 结构大致由 3 部分组成，它们分别是列名称、索引和数据。
![输入图片说明](/imgs/2024-05-07/2IOH1eC0IgwMSCSj.png)
Pandas 提供了 `head()` 和 `tail()` 方法，它可以帮助我们只预览一小块数据。

```python
df.head()  # 默认显示前 5 条
```

```python
df.tail(7)  # 指定显示后 7 条
```

Pandas 还提供了统计和描述性方法，方便你从宏观的角度去了解数据集。`describe()` 相当于对数据集进行概览，会输出该数据集每一列数据的计数、最大值、最小值等。

```python
df.describe()
```

Pandas 基于 NumPy 开发，所以任何时候你都可以通过 `.values` 将 DataFrame 转换为 NumPy 数组。

```python
df.values
```

这也就说明了，你可以同时使用 Pandas 和 NumPy 提供的 API 对同一数据进行操作，并在二者之间进行随意转换。这就是一个非常灵活的工具生态圈。

除了 `.values`，DataFrame 支持的常见属性可以通过 [_官方文档相应章节_](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#attributes-and-underlying-data) 查看。其中常用的有：

```python
df.index  # 查看索引
```

```python
df.columns  # 查看列名
```

```python
df.shape  # 查看形状
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2MDMyNDY0ODAsNDk3ODE4ODEwXX0=
-->