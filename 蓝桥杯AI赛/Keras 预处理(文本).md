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
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTIzNjExNzg1Nyw0NDA5MDU2MTldfQ==
-->