## Word2Vec
Word2Vec 是一种将词语转换为向量的技术，它能够捕捉词语之间的语义关系。通过这种方式，机器可以理解并处理文本数据。Word2Vec 有两种主要的模型：**CBOW (Continuous Bag of Words)** 和 **Skip-gram**。

### 示例用法

假设我们有一个文本语料库，我们希望使用 Word2Vec 将其中的词语转换为向量表示。以下是使用 Python 的 `gensim` 库来实现 Word2Vec 的示例。

### 示例用法

假设我们有一个文本语料库，我们希望使用 Word2Vec 将其中的词语转换为向量表示。以下是使用 Python 的 `gensim` 库来实现 Word2Vec 的示例。

1.  **安装依赖包**
    
    
~~~bash
pip install gensim
~~~
    
2.  **导入所需的库**
    
~~~py
from gensim.models import Word2Vec
~~~
    
3.  **准备训练数据** 通常，我们需要对文本进行预处理，包括分词等。假设我们有一个简单的句子列表作为我们的语料库：
    
~~~py
sentences = [     ["I", "love", "machine", "learning"],     ["Gensim", "is", "a", "great", "tool"],     ["Word2Vec", "transforms", "words", "to", "vectors"],     ["Natural", "language", "processing", "is", "fun"] ]
~~~
    
4.  **训练 Word2Vec 模型**
    
~~~py
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
~~~
    这里的参数解释：
    
    -   `vector_size`：词向量的维度。
    -   `window`：窗口大小，表示当前词语与预测词语之间的最大距离。
    -   `min_count`：忽略出现频率低于这个值的词语。
    -   `workers`：使用的线程数。
5.  **使用模型** 一旦模型训练完毕，我们可以使用它来获取词语的向量表示或寻找相似词语。例如：
    
~~~py
# 获取词语 "machine" 的向量表示 vector = model.wv['machine'] print(vector)  # 找到与 "machine" 最相似的词语 similar_words = model.wv.most_similar('machine', topn=5) print(similar_words)
~~~

### 示例解释

-   **获取词向量**：`model.wv['machine']` 返回词语 "machine" 的向量表示，这是一个包含 100 个元素的数组（因为我们设置了 `vector_size=100`）。
-   **寻找相似词语**：`model.wv.most_similar('machine', topn=5)` 返回与 "machine" 最相似的 5 个词语及其相似度分数。

### 完整代码示例

~~~py
from gensim.models import Word2Vec  
# 示例语料库 
sentences = [     ["I", "love", "machine", "learning"],     ["Gensim", "is", "a", "great", "tool"],     ["Word2Vec", "transforms", "words", "to", "vectors"],     ["Natural", "language", "processing", "is", "fun"] ]  
# 训练 Word2Vec 模型 
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)  
# 获取词语 "machine" 的向量表示 
vector = model.wv['machine'] 
print("Vector representation of 'machine':\n", vector)  
# 找到与 "machine" 最相似的词语 
similar_words = model.wv.most_similar('machine', topn=5) 
print("\nWords most similar to 'machine':") 
for word, similarity in similar_words:     
	print(f"{word}: {similarity:.4f}")
~~~
通过这个示例，您可以了解如何使用 Word2Vec 将词语转换为向量，并进行相似词语的查找。这种技术在自然语言处理任务中非常有用，例如文本分类、聚类、推荐系统等。

## jieba
`jieba` 是一个非常流行的中文分词工具，使用方便且功能强大。它支持三种分词模式：**精确模式**、**全模式** 和 **搜索引擎模式**。此外，`jieba` 还支持自定义词典和关键词提取。下面是一些常见用法的示例。
首先，确保安装了 `jieba` 库：
### 安装 jieba

~~~bash
pip install jieba
~~~
### 导入 jieba

~~~py
import jieba
~~~

### 1. 精确模式

这是 `jieba` 的默认模式，会精确地切分句子，不存在冗余。适合文本分析。


~~~py
text = "我爱自然语言处理" seg_list = jieba.cut(text, 
cut_all=False) print("精确模式:", "/ ".join(seg_list))
~~~

~~~makefile
精确模式: 我/ 爱/ 自然/ 语言/ 处理
~~~

### 2. 全模式

把句子中所有的可以成词的词语都扫描出来，有冗余。适合需要穷尽所有可能词语的情况。


~~~py
seg_list = jieba.cut(text, cut_all=True) print("全模式:", "/ ".join(seg_list))
~~~
输出：


~~~makefile
全模式: 我/ 爱/ 自然/ 自然语言/ 语言/ 处理
~~~

### 3. 搜索引擎模式

在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎构建索引。
~~~py
seg_list = jieba.cut_for_search(text) 
print("搜索引擎模式:", "/ ".join(seg_list))
~~~

~~~makefile
搜索引擎模式: 我/ 爱/ 自然/ 语言/ 自然语言/ 处理
~~~

### 4. 自定义词典

可以加载用户自定义的词典来补充 `jieba` 自带词典的不足。例如，我们希望把 "自然语言处理" 当作一个词来看待：


~~~py
jieba.load_userdict("mydict.txt")
~~~

`mydict.txt` 的内容可以是：


~~~
自然语言处理 10
~~~

其中 `10` 表示词频。

### 5. 关键词提取

`jieba.analyse` 模块提供了关键词提取功能。它可以通过 TF-IDF 算法提取关键词：
~~~py
from jieba import analyse
text = "我爱自然语言处理，特别是使用jieba进行分词。" 
keywords = analyse.extract_tags(text, topK=5, withWeight=False) 
print("关键词:", keywords)
~~~

输出：

~~~less

关键词: ['自然语言处理', '分词', '特别', '进行', '使用']`

### 6. 词性标注

`jieba.posseg` 模块用于词性标注，可以标注每个词语的词性：

py

复制代码

`import jieba.posseg as pseg  words = pseg.cut(text) for word, flag in words:     print(f"{word} ({flag})")`

输出：

scss

复制代码

`我 (r) 爱 (v) 自然语言处理 (nz) ， (x) 特别 (d) 是 (v) 使用 (v) jieba (eng) 进行 (v) 分词 (v) 。 (x)`

### 7. 并行分词

`jieba` 还支持并行分词，可以在多核 CPU 下加快分词速度：

py

复制代码

`jieba.enable_parallel(4)  # 开启并行分词，参数为并行进程数  seg_list = jieba.cut(text, cut_all=False) print("并行分词:", "/ ".join(seg_list))  jieba.disable_parallel()  # 关闭并行分词`

通过这些示例，您可以看到 `jieba` 的强大功能和灵活性，可以满足不同场景下的中文分词需求。
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTY0NjQ2MDUxMywxNTcwMDIyMjU0XX0=
-->