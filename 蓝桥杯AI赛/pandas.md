# 数据类型
## series

[_Series_](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html) 是 Pandas 中最基本的一维数组形式。其可以储存整数、浮点数、字符串等类型的数据。Series 基本结构如下：

```python
pandas.Series(data=None, index=None)
```

其中，`data` 可以是字典，或者NumPy 里的 ndarray 对象等。`index` 是数据索引，索引是 Pandas 数据结构中的一大特性，它主要的功能是帮助我们更快速地定位数据。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4ODgyNTU4NjMsNDk3ODE4ODEwXX0=
-->