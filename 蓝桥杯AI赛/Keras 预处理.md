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

### pad_sequences

```python
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
```

将多个序列截断或补齐为相同长度。

该函数将一个 `num_samples` 的序列（整数列表）转化为一个 2D Numpy 矩阵，其尺寸为 `(num_samples, num_timesteps)`。 `num_timesteps` 要么是给定的 `maxlen` 参数，要么是最长序列的长度。

比 `num_timesteps` 短的序列将在末端以 `value` 值补齐。

比 `num_timesteps` 长的序列将会被截断以满足所需要的长度。补齐或截断发生的位置分别由参数 `pading` 和 `truncating` 决定。

向前补齐为默认操作。

**参数**

-   **sequences**: 列表的列表，每一个元素是一个序列。
-   **maxlen**: 整数，所有序列的最大长度。
-   **dtype**: 输出序列的类型。 要使用可变长度字符串填充序列，可以使用 `object`。
-   **padding**: 字符串，'pre' 或 'post' ，在序列的前端补齐还是在后端补齐。
-   **truncating**: 字符串，'pre' 或 'post' ，移除长度大于 `maxlen` 的序列的值，要么在序列前端截断，要么在后端。
-   **value**: 浮点数，表示用来补齐的值。

**返回**

-   **x**: Numpy 矩阵，尺寸为 `(len(sequences), maxlen)`。

**异常**

-   ValueError: 如果截断或补齐的值无效，或者序列条目的形状无效。


`skipgrams` 是一种用于生成 Skip-gram 模型训练数据的方法。Skip-gram 模型通过预测上下文词来学习词向量。通过使用 Keras 的 `skipgrams` 函数，可以轻松地生成 (target_word, context_word) 对，用于训练神经网络模型，如 Word2Vec。这种技术在自然语言处理任务中广泛应用，用于捕捉词语之间的语义关系。



### make_sampling_table

```py
keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-05)
```

生成一个基于单词的概率采样表。

用来生成 `skipgrams` 的 `sampling_table` 参数。`sampling_table[i]` 是数据集中第 i 个最常见词的采样概率（出于平衡考虑，出现更频繁的词应该被更少地采样）。

采样概率根据 word2vec 中使用的采样分布生成：

```py
p(word) = (min(1, sqrt(word_frequency / sampling_factor) /
    (word_frequency / sampling_factor)))
```

我们假设单词频率遵循 Zipf 定律（s=1），来导出 frequency(rank) 的数值近似：

`frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`，其中 `gamma` 为 Euler-Mascheroni 常量。

**参数**

-   **size**: 整数，可能采样的单词数量。
-   **sampling_factor**: word2vec 公式中的采样因子。

**返回**

一个长度为 `size` 大小的 1D Numpy 数组，其中第 i 项是排名为 i 的单词的采样概率。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQwNzI5NTY5Miw2MjQ3MzA1NjEsMTQ0MT
UxOTQwOF19
-->