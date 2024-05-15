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

```python
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), 
padding='valid', data_format=None, dilation_rate=(1, 1), 
activation=None, use_bias=True, 
kernel_initializer='glorot_uniform', 
bias_initializer='zeros', kernel_regularizer=None, 
bias_regularizer=None, activity_regularizer=None, 
kernel_constraint=None, bias_constraint=None)
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


### MaxPooling1D

```python
keras.layers.MaxPooling1D(pool_size=2, strides=None, 
padding='valid', data_format='channels_last')
```

对于时序数据的最大池化。

**参数**

-   **pool_size**: 整数，最大池化的窗口大小。
-   **strides**: 整数，或者是 `None`。作为缩小比例的因数。 例如，2 会使得输入张量缩小一半。 如果是 `None`，那么默认值是 `pool_size`。
-   **padding**: `"valid"` 或者 `"same"` （区分大小写）。
-   **data_format**: 字符串，`channels_last` (默认)或 `channels_first` 之一。 表示输入各维度的顺序。 `channels_last` 对应输入尺寸为 `(batch, steps, features)`， `channels_first` 对应输入尺寸为 `(batch, features, steps)`。

**输入尺寸**

-   如果 `data_format='channels_last'`， 输入为 3D 张量，尺寸为： `(batch_size, steps, features)`
-   如果`data_format='channels_first'`， 输入为 3D 张量，尺寸为： `(batch_size, features, steps)`

**输出尺寸**

-   如果 `data_format='channels_last'`， 输出为 3D 张量，尺寸为： `(batch_size, downsampled_steps, features)`
-   如果 `data_format='channels_first'`， 输出为 3D 张量，尺寸为： `(batch_size, features, downsampled_steps)`

----------

[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L217)

### MaxPooling2D

```python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, 
padding='valid', data_format=None)
```

对于空间数据的最大池化。

**参数**

-   **pool_size**: 整数，或者 2 个整数表示的元组， 沿（垂直，水平）方向缩小比例的因数。 （2，2）会把输入张量的两个维度都缩小一半。 如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
-   **strides**: 整数，2 个整数表示的元组，或者是 `None`。 表示步长值。 如果是 `None`，那么默认值是 `pool_size`。
-   **padding**: `"valid"` 或者 `"same"` （区分大小写）。
-   **data_format**: 字符串，`channels_last` (默认)或 `channels_first` 之一。 表示输入各维度的顺序。 `channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量， 而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。 默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。 如果还没有设置过，那么默认值就是 "channels_last"。

**输入尺寸**

-   如果 `data_format='channels_last'`: 尺寸是 `(batch_size, rows, cols, channels)` 的 4D 张量
-   如果 `data_format='channels_first'`: 尺寸是 `(batch_size, channels, rows, cols)` 的 4D 张量

**输出尺寸**

-   如果 `data_format='channels_last'`: 尺寸是 `(batch_size, pooled_rows, pooled_cols, channels)` 的 4D 张量
-   如果 `data_format='channels_first'`: 尺寸是 `(batch_size, channels, pooled_rows, pooled_cols)` 的 4D 张量

----------

[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L386)

### MaxPooling3D

```python
keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, 
padding='valid', data_format=None)
```

对于 3D（空间，或时空间）数据的最大池化。

**参数**

-   **pool_size**: 3 个整数表示的元组，缩小（dim1，dim2，dim3）比例的因数。 (2, 2, 2) 会把 3D 输入张量的每个维度缩小一半。
-   **strides**: 3 个整数表示的元组，或者是 `None`。步长值。
-   **padding**: `"valid"` 或者 `"same"`（区分大小写）。
-   **data_format**: 字符串，`channels_last` (默认)或 `channels_first` 之一。 表示输入各维度的顺序。 `channels_last` 代表尺寸是 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的输入张量， 而 `channels_first` 代表尺寸是 `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的输入张量。 默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。 如果还没有设置过，那么默认值就是 "channels_last"。

**输入尺寸**

-   如果 `data_format='channels_last'`: 尺寸是 `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的 5D 张量
-   如果 `data_format='channels_first'`: 尺寸是 `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的 5D 张量

**输出尺寸**

-   如果 `data_format='channels_last'`: 尺寸是 `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)` 的 5D 张量
-   如果 `data_format='channels_first'`: 尺寸是 `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)` 的 5D 张量

池化层（Pooling Layer）是卷积神经网络（CNN）中的一种层类型，主要用于减少特征图的尺寸，从而降低计算复杂度，防止过拟合，并且提取特征的平移不变性。池化层有多种类型，最常用的包括最大池化（Max Pooling）和平均池化（Average Pooling）。

### AveragePooling1D

```python
keras.layers.AveragePooling1D(pool_size=2, strides=None, 
padding='valid', data_format='channels_last')
```

对于时序数据的平均池化。

**参数**

-   **pool_size**: 整数，平均池化的窗口大小。
-   **strides**: 整数，或者是 `None`。作为缩小比例的因数。 例如，2 会使得输入张量缩小一半。 如果是 `None`，那么默认值是 `pool_size`。
-   **padding**: `"valid"` 或者 `"same"` （区分大小写）。
-   **data_format**: 字符串，`channels_last` (默认)或 `channels_first` 之一。 表示输入各维度的顺序。 `channels_last` 对应输入尺寸为 `(batch, steps, features)`， `channels_first` 对应输入尺寸为 `(batch, features, steps)`。

**输入尺寸**

-   如果 `data_format='channels_last'`， 输入为 3D 张量，尺寸为： `(batch_size, steps, features)`
-   如果`data_format='channels_first'`， 输入为 3D 张量，尺寸为： `(batch_size, features, steps)`

**输出尺寸**

-   如果 `data_format='channels_last'`， 输出为 3D 张量，尺寸为： `(batch_size, downsampled_steps, features)`
-   如果 `data_format='channels_first'`， 输出为 3D 张量，尺寸为： `(batch_size, features, downsampled_steps)`

----------

[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L272)

### AveragePooling2D

```python
keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, 
padding='valid', data_format=None)
```

对于空间数据的平均池化。

**参数**

-   **pool_size**: 整数，或者 2 个整数表示的元组， 沿（垂直，水平）方向缩小比例的因数。 （2，2）会把输入张量的两个维度都缩小一半。 如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
-   **strides**: 整数，2 个整数表示的元组，或者是 `None`。 表示步长值。 如果是 `None`，那么默认值是 `pool_size`。
-   **padding**: `"valid"` 或者 `"same"` （区分大小写）。
-   **data_format**: 字符串，`channels_last` (默认)或 `channels_first` 之一。 表示输入各维度的顺序。 `channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量， 而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。 默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。 如果还没有设置过，那么默认值就是 "channels_last"。

**输入尺寸**

-   如果 `data_format='channels_last'`: 尺寸是 `(batch_size, rows, cols, channels)` 的 4D 张量
-   如果 `data_format='channels_first'`: 尺寸是 `(batch_size, channels, rows, cols)` 的 4D 张量

**输出尺寸**

-   如果 `data_format='channels_last'`: 尺寸是 `(batch_size, pooled_rows, pooled_cols, channels)` 的 4D 张量
-   如果 `data_format='channels_first'`: 尺寸是 `(batch_size, channels, pooled_rows, pooled_cols)` 的 4D 张量

----------

[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L436)

### AveragePooling3D

```python
keras.layers.AveragePooling3D(pool_size=(2, 2, 2), 
strides=None, padding='valid', data_format=None)
```

对于 3D （空间，或者时空间）数据的平均池化。

**参数**

-   **pool_size**: 3 个整数表示的元组，缩小（dim1，dim2，dim3）比例的因数。 (2, 2, 2) 会把 3D 输入张量的每个维度缩小一半。
-   **strides**: 3 个整数表示的元组，或者是 `None`。步长值。
-   **padding**: `"valid"` 或者 `"same"`（区分大小写）。
-   **data_format**: 字符串，`channels_last` (默认)或 `channels_first` 之一。 表示输入各维度的顺序。 `channels_last` 代表尺寸是 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的输入张量， 而 `channels_first` 代表尺寸是 `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的输入张量。 默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。 如果还没有设置过，那么默认值就是 "channels_last"。

**输入尺寸**

-   如果 `data_format='channels_last'`: 尺寸是 `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的 5D 张量
-   如果 `data_format='channels_first'`: 尺寸是 `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的 5D 张量

**输出尺寸**

-   如果 `data_format='channels_last'`: 尺寸是 `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)` 的 5D 张量
-   如果 `data_format='channels_first'`: 尺寸是 `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)` 的 5D 张量

----------

[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L557)

### GlobalMaxPooling1D

```
keras.layers.GlobalMaxPooling1D(data_format='channels_last')
```

对于时序数据的全局最大池化。

**参数**

-   **data_format**: 字符串，`channels_last` (默认)或 `channels_first` 之一。 表示输入各维度的顺序。 `channels_last` 对应输入尺寸为 `(batch, steps, features)`， `channels_first` 对应输入尺寸为 `(batch, features, steps)`。

**输入尺寸**

尺寸是 `(batch_size, steps, features)` 的 3D 张量。

**输出尺寸**

尺寸是 `(batch_size, features)` 的 2D 张量。

#### 主要作用

1.  **降维和减少计算量**：
    
    -   池化层通过对输入特征图进行下采样（通常是通过取局部区域的最大值或平均值），显著减少了特征图的尺寸，从而减少了后续层的计算量。
    -   例如，经过 2x2 的最大池化后，输入特征图的尺寸将减少到原来的一半。
2.  **防止过拟合**：
    
    -   池化层通过减少特征图的尺寸和参数数量，有助于减轻过拟合。
    -   池化操作具有一定的统计平稳性，使得模型更具泛化能力。
3.  **提取平移不变性**：
    
    -   由于池化层对局部区域进行操作，它可以使网络对输入图像的小幅度平移保持不变（即输入图像稍微移动时，特征图不会发生显著变化）。
    -   这使得模型对输入的变化更为鲁棒。



### RNN
[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L238)
```py
keras.layers.RNN(cell, return_sequences=False, 
return_state=False, go_backwards=False, stateful=False, 
unroll=False)
```

循环神经网络层基类。

**参数**

-   **cell**: 一个 RNN 单元实例。RNN 单元是一个具有以下几项的类：
    
    -   一个 `call(input_at_t, states_at_t)` 方法， 它返回 `(output_at_t, states_at_t_plus_1)`。 单元的调用方法也可以采引入可选参数 `constants`， 详见下面的小节「关于给 RNN 传递外部常量的说明」。
    -   一个 `state_size` 属性。这可以是单个整数（单个状态）， 在这种情况下，它是循环层状态的大小（应该与单元输出的大小相同）。 这也可以是整数表示的列表/元组（每个状态一个大小）。
    -   一个 `output_size` 属性。 这可以是单个整数或者是一个 TensorShape， 它表示输出的尺寸。出于向后兼容的原因，如果此属性对于当前单元不可用， 则该值将由 `state_size` 的第一个元素推断。
    
    `cell` 也可能是 RNN 单元实例的列表，在这种情况下，RNN 的单元将堆叠在另一个单元上，实现高效的堆叠 RNN。
    
-   **return_sequences**: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
    
-   **return_state**: 布尔值。除了输出之外是否返回最后一个状态。
-   **go_backwards**: 布尔值 (默认 False)。 如果为 True，则向后处理输入序列并返回相反的序列。
-   **stateful**: 布尔值 (默认 False)。 如果为 True，则批次中索引 i 处的每个样品的最后状态将用作下一批次中索引 i 样品的初始状态。
-   **unroll**: 布尔值 (默认 False)。 如果为 True，则网络将展开，否则将使用符号循环。 展开可以加速 RNN，但它往往会占用更多的内存。 展开只适用于短序列。
-   **input_dim**: 输入的维度（整数）。 将此层用作模型中的第一层时，此参数（或者，关键字参数 `input_shape`）是必需的。
-   **input_length**: 输入序列的长度，在恒定时指定。 如果你要在上游连接 `Flatten` 和 `Dense` 层， 则需要此参数（如果没有它，无法计算全连接输出的尺寸）。 请注意，如果循环神经网络层不是模型中的第一层， 则需要在第一层的层级指定输入长度（例如，通过 `input_shape` 参数）。

**输入尺寸**

3D 张量，尺寸为 `(batch_size, timesteps, input_dim)`。

**输出尺寸**

-   如果 `return_state`：返回张量列表。 第一个张量为输出。剩余的张量为最后的状态， 每个张量的尺寸为 `(batch_size, units)`。例如，对于 RNN/GRU，状态张量数目为 1，对 LSTM 为 2。
-   如果 `return_sequences`：返回 3D 张量， 尺寸为 `(batch_size, timesteps, units)`。
-   否则，返回尺寸为 `(batch_size, units)` 的 2D 张量。

**Masking**

该层支持以可变数量的时间步对输入数据进行 masking。 要将 masking 引入你的数据，请使用 [Embedding](https://keras-zh.readthedocs.io/layers/embeddings/) 层， 并将 `mask_zero` 参数设置为 `True`。

**关于在 RNN 中使用「状态（statefulness）」的说明**

你可以将 RNN 层设置为 `stateful`（有状态的）， 这意味着针对一个批次的样本计算的状态将被重新用作下一批样本的初始状态。 这假定在不同连续批次的样品之间有一对一的映射。

为了使状态有效：

-   在层构造器中指定 `stateful=True`。
-   为你的模型指定一个固定的批次大小， 如果是顺序模型，为你的模型的第一层传递一个 `batch_input_shape=(...)` 参数。
-   为你的模型指定一个固定的批次大小， 如果是顺序模型，为你的模型的第一层传递一个 `batch_input_shape=(...)`。 如果是带有 1 个或多个 Input 层的函数式模型，为你的模型的所有第一层传递一个 `batch_shape=(...)`。 这是你的输入的预期尺寸，_包括批量维度_。 它应该是整数的元组，例如 `(32, 10, 100)`。
-   在调用 `fit()` 是指定 `shuffle=False`。

要重置模型的状态，请在特定图层或整个模型上调用 `.reset_states()`。

**关于指定 RNN 初始状态的说明**

您可以通过使用关键字参数 `initial_state` 调用它们来符号化地指定 RNN 层的初始状态。 `initial_state` 的值应该是表示 RNN 层初始状态的张量或张量列表。

您可以通过调用带有关键字参数 `states` 的 `reset_states` 方法来数字化地指定 RNN 层的初始状态。 `states` 的值应该是一个代表 RNN 层初始状态的 Numpy 数组或者 Numpy 数组列表。

**关于给 RNN 传递外部常量的说明**

你可以使用 `RNN.__call__`（以及 `RNN.call`）的 `constants` 关键字参数将「外部」常量传递给单元。 这要求 `cell.call` 方法接受相同的关键字参数 `constants`。 这些常数可用于调节附加静态输入（不随时间变化）上的单元转换，也可用于注意力机制。

**示例**

```py
# 首先，让我们定义一个 RNN 单元，作为网络层子类。

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# 让我们在 RNN 层使用这个单元：

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# 以下是如何使用单元格构建堆叠的 RNN的方法：

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
```
### RNN的基本原理

RNN具有一个循环结构，这使得它能够保持一个“状态”，并通过这个状态来处理序列中的每一个元素。具体来说，对于每一个时间步，RNN会根据当前输入和前一个时间步的状态来更新当前的状态，并产生一个输出。

#### 公式

对于一个简单的RNN：

-   𝑥𝑡xt​：在时间步 𝑡t 的输入
-   ℎ𝑡ht​：在时间步 𝑡t 的隐藏状态（或记忆）
-   𝑦𝑡yt​：在时间步 𝑡t 的输出
-   𝑊W：输入到隐藏层的权重矩阵
-   𝑈U：隐藏层到隐藏层的权重矩阵
-   𝑉V：隐藏层到输出的权重矩阵
-   𝑏b、𝑐c：偏置项
-   𝜎σ：激活函数（如tanh或ReLU）

隐藏状态和输出的更新公式如下：

1.  **隐藏状态更新**： ℎ𝑡=𝜎(𝑊𝑥𝑡+𝑈ℎ𝑡−1+𝑏)ht​=σ(Wxt​+Uht−1​+b)
    
2.  **输出**： 𝑦𝑡=𝑉ℎ𝑡+𝑐yt​=Vht​+c
    

### RNN的特性

1.  **共享参数**：在所有时间步之间共享相同的权重矩阵 𝑊W、𝑈U 和 𝑉V，这使得RNN可以处理任意长度的序列。
2.  **时间相关性**：通过隐藏状态 ℎ𝑡ht​ 来传递信息，使得前面时间步的信息能够影响后续时间步的输出。

### RNN的优缺点

#### 优点

-   **处理序列数据**：RNN特别适用于处理序列数据，如时间序列预测、自然语言处理和语音识别等。
-   **可变长度输入**：RNN能够处理不同长度的输入序列，而不需要对数据进行填充或裁剪。

#### 缺点

-   **梯度消失和爆炸问题**：在反向传播过程中，随着时间步数增加，梯度可能会消失或爆炸，导致训练过程变得困难。
-   **长期依赖问题**：RNN在捕捉长时间依赖关系时效果不佳，因为远距离的信息可能会被遗忘。

### 改进的RNN结构

为了克服RNN的缺点，研究人员提出了几种改进的RNN结构：

#### 长短期记忆网络（LSTM）

LSTM通过引入三个门（输入门、遗忘门和输出门）来控制信息的流动，从而解决了梯度消失和长期依赖问题。

-   **输入门**：控制当前输入的信息有多少传递到细胞状态。
-   **遗忘门**：控制前一个时间步的状态有多少保留到当前时间步。
-   **输出门**：控制细胞状态的哪一部分输出。

#### 门控循环单元（GRU）

GRU是LSTM的简化版本，它通过两个门（重置门和更新门）来控制信息的流动，同样能够有效地处理长时间依赖关系。
----------

[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L947)

### SimpleRNN

```py
keras.layers.SimpleRNN(units, activation='tanh', 
use_bias=True, kernel_initializer='glorot_uniform', 
recurrent_initializer='orthogonal', bias_initializer='zeros', 
kernel_regularizer=None, recurrent_regularizer=None, 
bias_regularizer=None, activity_regularizer=None, 
kernel_constraint=None, recurrent_constraint=None, 
bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, 
return_sequences=False, return_state=False, 
go_backwards=False, stateful=False, unroll=False)
```

全连接的 RNN，其输出将被反馈到输入。

**参数**

-   **units**: 正整数，输出空间的维度。
-   **activation**: 要使用的激活函数 (详见 [activations](https://keras-zh.readthedocs.io/activations/))。 默认：双曲正切（`tanh`）。 如果传入 `None`，则不使用激活函数 (即 线性激活：`a(x) = x`)。
-   **use_bias**: 布尔值，该层是否使用偏置向量。
-   **kernel_initializer**: `kernel` 权值矩阵的初始化器， 用于输入的线性转换 (详见 [initializers](https://keras-zh.readthedocs.io/initializers/))。
-   **recurrent_initializer**: `recurrent_kernel` 权值矩阵 的初始化器，用于循环层状态的线性转换 (详见 [initializers](https://keras-zh.readthedocs.io/initializers/))。
-   **bias_initializer**:偏置向量的初始化器 (详见[initializers](https://keras-zh.readthedocs.io/initializers/)).
-   **kernel_regularizer**: 运用到 `kernel` 权值矩阵的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **recurrent_regularizer**: 运用到 `recurrent_kernel` 权值矩阵的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **bias_regularizer**: 运用到偏置向量的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **activity_regularizer**: 运用到层输出（它的激活值）的正则化函数 (详见 [regularizer](https://keras-zh.readthedocs.io/regularizers/))。
-   **kernel_constraint**: 运用到 `kernel` 权值矩阵的约束函数 (详见 [constraints](https://keras-zh.readthedocs.io/constraints/))。
-   **recurrent_constraint**: 运用到 `recurrent_kernel` 权值矩阵的约束函数 (详见 [constraints](https://keras-zh.readthedocs.io/constraints/))。
-   **bias_constraint**: 运用到偏置向量的约束函数 (详见 [constraints](https://keras-zh.readthedocs.io/constraints/))。
-   **dropout**: 在 0 和 1 之间的浮点数。 单元的丢弃比例，用于输入的线性转换。
-   **recurrent_dropout**: 在 0 和 1 之间的浮点数。 单元的丢弃比例，用于循环层状态的线性转换。
-   **return_sequences**: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
-   **return_state**: 布尔值。除了输出之外是否返回最后一个状态。
-   **go_backwards**: 布尔值 (默认 False)。 如果为 True，则向后处理输入序列并返回相反的序列。
-   **stateful**: 布尔值 (默认 False)。 如果为 True，则批次中索引 i 处的每个样品 的最后状态将用作下一批次中索引 i 样品的初始状态。
-   **unroll**: 布尔值 (默认 False)。 如果为 True，则网络将展开，否则将使用符号循环。 展开可以加速 RNN，但它往往会占用更多的内存。 展开只适用于短序列。
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTQzMDQwNjUzMSw3MTI2MTgwMDgsODU2OT
cyMDQzLDIxMDgxMTA1MzBdfQ==
-->