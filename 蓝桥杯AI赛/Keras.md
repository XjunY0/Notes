# Keras Sequential 顺序模型

顺序模型是多个网络层的线性堆叠。

你可以通过将网络层实例的列表传递给 `Sequential` 的构造器，来创建一个 `Sequential` 模型：

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```

也可以简单地使用 `.add()` 方法将各层添加到模型中：

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```

----------

## 指定输入数据的尺寸

模型需要知道它所期望的输入的尺寸。出于
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgxMDQyOTE3LC0yMDg4NzQ2NjEyXX0=
-->