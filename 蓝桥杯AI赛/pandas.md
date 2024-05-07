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

可以有三种方式创建一个
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4OTMyMzMyODIsNDk3ODE4ODEwXX0=
-->