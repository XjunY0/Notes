### TimeseriesGenerator

```py
keras.preprocessing.sequence.TimeseriesGenerator(data, targets, length, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False, reverse=False, batch_size=128)
```

用于生成批量时序数据的实用工具类。

这个类以一系列由相等间隔以及一些时间序列参数（例如步长、历史长度等）汇集的数据点作为输入，以生成用于训练/验证的批次数据。

**参数**

-   **data**: 可索引的生成器（例如列表或 Numpy 数组），包含连续数据点（时间步）。数据应该是 2D 的，且第 0 个轴为时间维度。
-   **targets**: 对应于 `data` 的时间步的目标值。它应该与 `data` 的长度相同。
-   **length**: 输出序列的长度（以时间步数表示）。
-   **sampling_rate**: 序列内连续各个时间步之间的周期。对于周期 `r`, 时间步 `data[i]`, `data[i-r]`, ... `data[i - length]` 被用于生成样本序列。
-   **stride**: 连续输出序列之间的周期. 对于周期 `s`, 连续输出样本将为 `data[i]`, `data[i+s]`, `data[i+2*s]` 等。
-   **start_index**: 在 `start_index` 之前的数据点在输出序列中将不被使用。这对保留部分数据以进行测试或验证很有用。
-   **end_index**: 在 `end_index` 之后的数据点在输出序列中将不被使用。这对保留部分数据以进行测试或验证很有用。
-   **shuffle**: 是否打乱输出样本，还是按照时间顺序绘制它们。
-   **reverse**: 布尔值: 如果 `true`, 每个输出样本中的时间步将按照时间倒序排列。
-   **batch_size**: 每个批次中的时间序列样本数（可能除最后一个外）。

**返回**

一个 [Sequence](https://keras.io/zh/utils/#sequence) 实例。

**示例**

```py
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

data = np.array([[i] for i in range(50)])
targets = np.array([[i] for i in range(50)])

data_gen = TimeseriesGenerator(data, targets,
                               length=10, sampling_rate=2,
                               batch_size=2)
assert len(data_gen) == 20

batch_0 = data_gen[0]
x, y = batch_0
assert np.array_equal(x,
                      np.array([[[0], [2], [4], [6], [8]],
                                [[1], [3], [5], [7], [9]]]))
assert np.array_equal(y,
                      np.array([[10], [11]]))
```


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2ODI2NDkwMzZdfQ==
-->