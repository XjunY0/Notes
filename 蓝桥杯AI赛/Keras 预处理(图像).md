# 图像预处理

[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L238)

## ImageDataGenerator 类

```py
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, 
                                             samplewise_center=False, 
                                             featurewise_std_normalization=False, 
                                             samplewise_std_normalization=False, 
                                             zca_whitening=False, 
                                             zca_epsilon=1e-06, 
                                             rotation_range=0, 
                                             width_shift_range=0.0, 
                                             height_shift_range=0.0, 
                                             brightness_range=None, 
                                             shear_range=0.0, 
                                             zoom_range=0.0, 
                                             channel_shift_range=0.0, 
                                             fill_mode='nearest', 
                                             cval=0.0, 
                                             horizontal_flip=False, 
                                             vertical_flip=False, 
                                             rescale=None, 
                                             preprocessing_function=None, 
                                             data_format='channels_last', 
                                             validation_split=0.0, 
                                             interpolation_order=1, 
                                             dtype='float32')
```

通过实时数据增强生成张量图像数据批次。数据将不断循环（按批次）。

**参数**

-   **featurewise_center**: 布尔值。将输入数据的均值设置为 0，逐特征进行。
-   **samplewise_center**: 布尔值。将每个样本的均值设置为 0。
-   **featurewise_std_normalization**: Boolean. 布尔值。将输入除以数据标准差，逐特征进行。
-   **samplewise_std_normalization**: 布尔值。将每个输入除以其标准差。
-   **zca_epsilon**: ZCA 白化的 epsilon 值，默认为 1e-6。
-   **zca_whitening**: 布尔值。是否应用 ZCA 白化。
-   **rotation_range**: 整数。随机旋转的度数范围。
-   **width_shift_range**: 浮点数、一维数组或整数
    -   float: 如果 <1，则是除以总宽度的值，或者如果 >=1，则为像素值。
    -   1-D 数组: 数组中的随机元素。
    -   int: 来自间隔 `(-width_shift_range, +width_shift_range)` 之间的整数个像素。
    -   `width_shift_range=2` 时，可能值是整数 `[-1, 0, +1]`，与 `width_shift_range=[-1, 0, +1]` 相同；而 `width_shift_range=1.0` 时，可能值是 `[-1.0, +1.0)` 之间的浮点数。
-   **height_shift_range**: 浮点数、一维数组或整数
    -   float: 如果 <1，则是除以总宽度的值，或者如果 >=1，则为像素值。
    -   1-D array-like: 数组中的随机元素。
    -   int: 来自间隔 `(-height_shift_range, +height_shift_range)` 之间的整数个像素。
    -   `height_shift_range=2` 时，可能值是整数 `[-1, 0, +1]`，与 `height_shift_range=[-1, 0, +1]` 相同；而 `height_shift_range=1.0` 时，可能值是 `[-1.0, +1.0)` 之间的浮点数。
-   **brightness_range**: 两个浮点数的元组或列表。从中选择亮度偏移值的范围。
-   **shear_range**: 浮点数。剪切强度（以弧度逆时针方向剪切角度）。
-   **zoom_range**: 浮点数 或 `[lower, upper]`。随机缩放范围。如果是浮点数，`[lower, upper] = [1-zoom_range, 1+zoom_range]`。
-   **channel_shift_range**: 浮点数。随机通道转换的范围。
-   **fill_mode**: {"constant", "nearest", "reflect" or "wrap"} 之一。默认为 'nearest'。输入边界以外的点根据给定的模式填充：
    -   'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
    -   'nearest': aaaaaaaa|abcd|dddddddd
    -   'reflect': abcddcba|abcd|dcbaabcd
    -   'wrap': abcdabcd|abcd|abcdabcd
-   **cval**: 浮点数或整数。用于边界之外的点的值，当 `fill_mode = "constant"` 时。
-   **horizontal_flip**: 布尔值。随机水平翻转。
-   **vertical_flip**: 布尔值。随机垂直翻转。
-   **rescale**: 重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）。
-   **preprocessing_function**: 应用于每个输入的函数。这个函数会在任何其他改变之前运行。这个函数需要一个参数：一张图像（秩为 3 的 Numpy 张量），并且应该输出一个同尺寸的 Numpy 张量。
-   **data_format**: 图像数据格式，{"channels_first", "channels_last"} 之一。"channels_last" 模式表示图像输入尺寸应该为 `(samples, height, width, channels)`，"channels_first" 模式表示输入尺寸应该为 `(samples, channels, height, width)`。默认为 在 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值。如果你从未设置它，那它就是 "channels_last"。
-   **validation_split**: 浮点数。Float. 保留用于验证的图像的比例（严格在0和1之间）。
-   **dtype**: 生成数组使用的数据类型。

**示例**

使用 `.flow(x, y)` 的例子：

```py
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# 计算特征归一化所需的数量
# （如果应用 ZCA 白化，将计算标准差，均值，主成分）
datagen.fit(x_train)

# 使用实时数据增益的批数据对模型进行拟合：
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# 这里有一个更 「手动」的例子
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # 我们需要手动打破循环，
            # 因为生成器会无限循环
            break
```

使用 `.flow_from_directory(directory)` 的例子：

```
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```

同时转换图像和蒙版 (mask) 的例子。

```
# 创建两个相同参数的实例
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# 为 fit 和 flow 函数提供相同的种子和关键字参数
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# 将生成器组合成一个产生图像和蒙版（mask）的生成器
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```

使用 `.flow_from_dataframe(dataframe, directory` 的例子:

```

train_df = pandas.read_csv("./train.csv")
valid_df = pandas.read_csv("./valid.csv")

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='data/train',
        x_col="filename",
        y_col="class",
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory='data/validation',
        x_col="filename",
        y_col="class",
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTE4MjA3NDM1OF19
-->