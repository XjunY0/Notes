### Text Preprocessing

[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/text.py#L139)

### Tokenizer

```py
keras.preprocessing.text.Tokenizer(num_words=None, 
                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
                                   lower=True, 
                                   split=' ', 
                                   char_level=False, 
                                   oov_token=None, 
                                   document_count=0)
```

文本标记实用类。

该类允许使用两种方法向量化一个文本语料库： 将每个文本转化为一个整数序列（每个整数都是词典中标记的索引）； 或者将其转化为一个向量，其中每个标记的系数可以是二进制值、词频、TF-IDF 权重等。

**参数**

-   **num_words**: 需要保留的最大词数，基于词频。只有最常出现的 `num_words-1` 词会被保留。
-   **filters**: 一个字符串，其中每个元素是一个将从文本中过滤掉的字符。默认值是所有标点符号，加上制表符和换行符，减去 `'` 字符。
-   **lower**: 布尔值。是否将文本转换为小写。
-   **split**: 字符串。按该字符串切割文本。
-   **char_level**: 如果为 True，则每个字符都将被视为标记。
-   **oov_token**: 如果给出，它将被添加到 word_index 中，并用于在 `text_to_sequence` 调用期间替换词汇表外的单词。

默认情况下，删除所有标点符号，将文本转换为空格分隔的单词序列（单词可能包含 `'` 字符）。 这些序列然后被分割成标记列表。然后它们将被索引或向量化。

`0` 是不会被分配给任何单词的保留索引。

示例：
~~~python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(news_train + news_test)
X_train = tokenizer.texts_to_sequences(news_train)
X_test = tokenizer.texts_to_sequences(news_test)
~~~


### hashing_trick

```py
keras.preprocessing.text.hashing_trick(text, 
                                       n, 
                                       hash_function=None, 
                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
                                       lower=True, 
                                       split=' ')
```

将文本转换为固定大小散列空间中的索引序列。

**参数**

-   **text**: 输入文本（字符串）。
-   **n**: 散列空间维度。
-   **hash_function**: 默认为 python 散列函数，可以是 'md5' 或任意接受输入字符串并返回整数的函数。注意 'hash' 不是稳定的散列函数，所以它在不同的运行中不一致，而 'md5' 是一个稳定的散列函数。
-   **filters**: 要过滤的字符列表（或连接），如标点符号。默认：``!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n``，包含基本标点符号，制表符和换行符。
-   **lower**: 布尔值。是否将文本转换为小写。
-   **split**: 字符串。按该字符串切割文本。

**返回**

整数词索引列表（唯一性无法保证）。

`0` 是不会被分配给任何单词的保留索引。

由于哈希函数可能发生冲突，可能会将两个或更多字分配给同一索引。 碰撞的[概率](https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)与散列空间的维度和不同对象的数量有关。


### one_hot

```py
keras.preprocessing.text.one_hot(text, 
                                 n, 
                                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
                                 lower=True, 
                                 split=' ')
```

One-hot 将文本编码为大小为 n 的单词索引列表。

这是 `hashing_trick` 函数的一个封装， 使用 `hash` 作为散列函数；单词索引映射无保证唯一性。

**参数**

-   **text**: 输入文本（字符串）。
-   **n**: 整数。词汇表尺寸。
-   **filters**: 要过滤的字符列表（或连接），如标点符号。默认：``!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n``，包含基本标点符号，制表符和换行符。
-   **lower**: 布尔值。是否将文本转换为小写。
-   **split**: 字符串。按该字符串切割文本。

**返回**

[1, n] 之间的整数列表。每个整数编码一个词（唯一性无法保证）。



### text_to_word_sequence

```py
keras.preprocessing.text.text_to_word_sequence(text, 
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
                                               lower=True, 
                                               split=' ')
```

将文本转换为单词（或标记）的序列。

**参数**

-   **text**: 输入文本（字符串）。
-   **filters**: 要过滤的字符列表（或连接），如标点符号。默认：``!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n``，包含基本标点符号，制表符和换行符。
-   **lower**: 布尔值。是否将文本转换为小写。
-   **split**: 字符串。按该字符串切割文本。

**返回**

词或标记的列表。
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTQ2MTU1MTE2LDQ0MDkwNTYxOV19
-->