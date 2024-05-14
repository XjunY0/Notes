### Dense

```python
keras.layers.Dense(units, activation=None, use_bias=True, 
kernel_initializer='glorot_uniform', 
bias_initializer='zeros', kernel_regularizer=None, 
bias_regularizer=None, activity_regularizer=None, 
kernel_constraint=None, bias_constraint=None)
```

就是你常用的的全连接层。

`Dense` 实现以下操作：`output = activation(dot(input, kernel) + bias)` 其中 `activation` 是按逐个元素计算的激活函数，`kernel` 是由网络层创建的权值矩阵，以及 `bias` 是其创建的偏置向量 (只在 `use_bias` 为 `True` 时才有用)。

-   **注意**: 如果该层的输入的秩大于 2，那么它首先被展平然后 再计算与 `kernel` 的点乘。

**示例**

```python
# 作为 Sequential 模型的第一层
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# 现在模型就会以尺寸为 (*, 16) 的数组作为输入，
# 其输出数组的尺寸为 (*, 32)

# 在第一层之后，你就不再需要指定输入的尺寸了：
model.add(Dense(32))
```

**参数**

-   **units**: 正整数，输出空间维度。
-   **activation**: 激活函数 (详见 [activations](https://keras-zh.readthedocs.io/activations/))。 若不指定，则不使用激活函数 (即，线性激活: `a(x) = x`)。
-   **use_bias**: 布尔值，该层是否使用偏置向量。
-   **kernel_initializer**: `kernel` 权值矩阵的初始化器 (详见 [initializers](https://keras-zh.readthedocs.io/initializers/))。
-   **bias_initializer**: 偏置向量的初始化器 (详见 [initializers](https://keras-zh.readthedocs.io/initializers/))。
-   **kernel_regularizer**: 运用到 `kernel` 权值矩阵的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **bias_regularizer**: 运用到偏置向量的的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **activity_regularizer**: 运用到层的输出的正则化函数 (它的 "activation")。 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **kernel_constraint**: 运用到 `kernel` 权值矩阵的约束函数 (详见 [constraints](https://keras-zh.readthedocs.io/constraints/))。
-   **bias_constraint**: 运用到偏置向量的约束函数 (详见 [constraints](https://keras-zh.readthedocs.io/constraints/))。

**输入尺寸**

nD 张量，尺寸: `(batch_size, ..., input_dim)`。 最常见的情况是一个尺寸为 `(batch_size, input_dim)` 的 2D 输入。

**输出尺寸**

nD 张量，尺寸: `(batch_size, ..., units)`。 例如，对于尺寸为 `(batch_size, input_dim)` 的 2D 输入， 输出的尺寸为 `(batch_size, units)`。

----------

[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L277)

### Activation

```python
keras.layers.Activation(activation)
```

将激活函数应用于输出。

**参数**

-   **activation**: 要使用的激活函数的名称 (详见: [activations](https://keras-zh.readthedocs.io/activations/))， 或者选择一个 Theano 或 TensorFlow 操作。

**输入尺寸**

任意尺寸。 当使用此层作为模型中的第一层时， 使用参数 `input_shape` （整数元组，不包括样本数的轴）。

**输出尺寸**

与输入相同。


### Conv1D

```python
keras.layers.Conv1D(filters, kernel_size, strides=1, 
padding='valid', data_format='channels_last', 
dilation_rate=1, activation=None, use_bias=True, 
kernel_initializer='glorot_uniform', 
bias_initializer='zeros', kernel_regularizer=None, 
bias_regularizer=None, activity_regularizer=None, 
kernel_constraint=None, bias_constraint=None)
```

1D 卷积层 (例如时序卷积)。

该层创建了一个卷积核，该卷积核以 单个空间（或时间）维上的层输入进行卷积， 以生成输出张量。 如果 `use_bias` 为 True， 则会创建一个偏置向量并将其添加到输出中。 最后，如果 `activation` 不是 `None`，它也会应用于输出。

当使用该层作为模型第一层时，需要提供 `input_shape` 参数（整数元组或 `None`，不包含 batch 轴）， 例如，`input_shape=(10, 128)` 在 `data_format="channels_last"` 时表示 10 个 128 维的向量组成的向量序列， `(None, 128)` 表示每步 128 维的向量组成的变长序列。

**参数**

-   **filters**: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。
-   **kernel_size**: 一个整数，或者单个整数表示的元组或列表， 指明 1D 卷积窗口的长度。
-   **strides**: 一个整数，或者单个整数表示的元组或列表， 指明卷积的步长。 指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
-   **padding**: `"valid"`, `"causal"` 或 `"same"` 之一 (大小写敏感) `"valid"` 表示「不填充」。 `"same"` 表示填充输入以使输出具有与原始输入相同的长度。 `"causal"` 表示因果（膨胀）卷积， 例如，`output[t]` 不依赖于 `input[t+1:]`， 在模型不应违反时间顺序的时间数据建模时非常有用。 详见 [WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499)。
-   **data_format**: 字符串, `"channels_last"` (默认) 或 `"channels_first"` 之一。输入的各个维度顺序。 `"channels_last"` 对应输入尺寸为 `(batch, steps, channels)` (Keras 中时序数据的默认格式) 而 `"channels_first"` 对应输入尺寸为 `(batch, channels, steps)`。
-   **dilation_rate**: 一个整数，或者单个整数表示的元组或列表，指定用于膨胀卷积的膨胀率。 当前，指定任何 `dilation_rate` 值 != 1 与指定 stride 值 != 1 两者不兼容。
-   **activation**: 要使用的激活函数 (详见 [activations](https://keras-zh.readthedocs.io/activations/))。 如未指定，则不使用激活函数 (即线性激活： `a(x) = x`)。
-   **use_bias**: 布尔值，该层是否使用偏置向量。
-   **kernel_initializer**: `kernel` 权值矩阵的初始化器 (详见 [initializers](https://keras-zh.readthedocs.io/initializers/))。
-   **bias_initializer**: 偏置向量的初始化器 (详见 [initializers](https://keras-zh.readthedocs.io/initializers/))。
-   **kernel_regularizer**: 运用到 `kernel` 权值矩阵的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **bias_regularizer**: 运用到偏置向量的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **activity_regularizer**: 运用到层输出（它的激活值）的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **kernel_constraint**: 运用到 `kernel` 权值矩阵的约束函数 (详见 [constraints](https://keras-zh.readthedocs.io/constraints/))。
-   **bias_constraint**: 运用到偏置向量的约束函数 (详见 [constraints](https://keras-zh.readthedocs.io/constraints/))。

**输入尺寸**

3D 张量 ，尺寸为 `(batch_size, steps, input_dim)`。

**输出尺寸**

3D 张量，尺寸为 `(batch_size, new_steps, filters)`。 由于填充或窗口按步长滑动，`steps` 值可能已更改。

----------

[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L361)

### Conv2D

```
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2D 卷积层 (例如对图像的空间卷积)。

该层创建了一个卷积核， 该卷积核对层输入进行卷积， 以生成输出张量。 如果 `use_bias` 为 True， 则会创建一个偏置向量并将其添加到输出中。 最后，如果 `activation` 不是 `None`，它也会应用于输出。

当使用该层作为模型第一层时，需要提供 `input_shape` 参数 （整数元组，不包含 batch 轴），例如， `input_shape=(128, 128, 3)` 表示 128x128 RGB 图像， 在 `data_format="channels_last"` 时。

**参数**

-   **filters**: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。
-   **kernel_size**: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，为所有空间维度指定相同的值。
-   **strides**: 一个整数，或者 2 个整数表示的元组或列表， 指明卷积沿宽度和高度方向的步长。 可以是一个整数，为所有空间维度指定相同的值。 指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
-   **padding**: `"valid"` 或 `"same"` (大小写敏感)。
-   **data_format**: 字符串， `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。 `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`， `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。 它默认为从 Keras 配置文件 `~/.keras/keras.json` 中 找到的 `image_data_format` 值。 如果你从未设置它，将使用 `channels_last`。
-   **dilation_rate**: 一个整数或 2 个整数的元组或列表， 指定膨胀卷积的膨胀率。 可以是一个整数，为所有空间维度指定相同的值。 当前，指定任何 `dilation_rate` 值 != 1 与 指定 stride 值 != 1 两者不兼容。
-   **activation**: 要使用的激活函数 (详见 [activations](https://keras-zh.readthedocs.io/activations/))。 如果你不指定，则不使用激活函数 (即线性激活： `a(x) = x`)。
-   **use_bias**: 布尔值，该层是否使用偏置向量。
-   **kernel_initializer**: `kernel` 权值矩阵的初始化器 (详见 [initializers](https://keras-zh.readthedocs.io/initializers/))。
-   **bias_initializer**: 偏置向量的初始化器 (详见 [initializers](https://keras-zh.readthedocs.io/initializers/))。
-   **kernel_regularizer**: 运用到 `kernel` 权值矩阵的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **bias_regularizer**: 运用到偏置向量的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **activity_regularizer**: 运用到层输出（它的激活值）的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **kernel_constraint**: 运用到 `kernel` 权值矩阵的约束函数 (详见 [constraints](https://keras-zh.readthedocs.io/constraints/))。
-   **bias_constraint**: 运用到偏置向量的约束函数 (详见 [constraints](https://keras-zh.readthedocs.io/constraints/))。

**输入尺寸**

-   如果 data_format='channels_first'， 输入 4D 张量，尺寸为 `(samples, channels, rows, cols)`。
-   如果 data_format='channels_last'， 输入 4D 张量，尺寸为 `(samples, rows, cols, channels)`。

**输出尺寸**

-   如果 data_format='channels_first'， 输出 4D 张量，尺寸为 `(samples, filters, new_rows, new_cols)`。
-   如果 data_format='channels_last'， 输出 4D 张量，尺寸为 `(samples, new_rows, new_cols, filters)`。

由于填充的原因，`rows` 和 `cols` 值可能已更改。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI3OTk2NzAyOV19
-->