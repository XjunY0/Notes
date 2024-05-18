Word2Vec 是一种将词语转换为向量的技术，它能够捕捉词语之间的语义关系。通过这种方式，机器可以理解并处理文本数据。Word2Vec 有两种主要的模型：**CBOW (Continuous Bag of Words)** 和 **Skip-gram**。

### 示例用法

假设我们有一个文本语料库，我们希望使用 Word2Vec 将其中的词语转换为向量表示。以下是使用 Python 的 `gensim` 库来实现 Word2Vec 的示例。

### 示例用法

假设我们有一个文本语料库，我们希望使用 Word2Vec 将其中的词语转换为向量表示。以下是使用 Python 的 `gensim` 库来实现 Word2Vec 的示例。

1.  **安装依赖包**
    
    
~~~pip install gensim~~~
    
2.  **导入所需的库**
    
    python
    
    复制代码
    
    `from gensim.models import Word2Vec`
    
3.  **准备训练数据** 通常，我们需要对文本进行预处理，包括分词等。假设我们有一个简单的句子列表作为我们的语料库：
    
    python
    
    复制代码
    
    `sentences = [     ["I", "love", "machine", "learning"],     ["Gensim", "is", "a", "great", "tool"],     ["Word2Vec", "transforms", "words", "to", "vectors"],     ["Natural", "language", "processing", "is", "fun"] ]`
    
4.  **训练 Word2Vec 模型**
    
    python
    
    复制代码
    
    `model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)`
    
    这里的参数解释：
    
    -   `vector_size`：词向量的维度。
    -   `window`：窗口大小，表示当前词语与预测词语之间的最大距离。
    -   `min_count`：忽略出现频率低于这个值的词语。
    -   `workers`：使用的线程数。
5.  **使用模型** 一旦模型训练完毕，我们可以使用它来获取词语的向量表示或寻找相似词语。例如：
    
    python
    
    复制代码
    
    `# 获取词语 "machine" 的向量表示 vector = model.wv['machine'] print(vector)  # 找到与 "machine" 最相似的词语 similar_words = model.wv.most_similar('machine', topn=5) print(similar_words)`
    

### 示例解释

-   **获取词向量**：`model.wv['machine']` 返回词语 "machine" 的向量表示，这是一个包含 100 个元素的数组（因为我们设置了 `vector_size=100`）。
-   **寻找相似词语**：`model.wv.most_similar('machine', topn=5)` 返回与 "machine" 最相似的 5 个词语及其相似度分数。

### 完整代码示例

python

复制代码

`from gensim.models import Word2Vec  # 示例语料库 sentences = [     ["I", "love", "machine", "learning"],     ["Gensim", "is", "a", "great", "tool"],     ["Word2Vec", "transforms", "words", "to", "vectors"],     ["Natural", "language", "processing", "is", "fun"] ]  # 训练 Word2Vec 模型 model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)  # 获取词语 "machine" 的向量表示 vector = model.wv['machine'] print("Vector representation of 'machine':\n", vector)  # 找到与 "machine" 最相似的词语 similar_words = model.wv.most_similar('machine', topn=5) print("\nWords most similar to 'machine':") for word, similarity in similar_words:     print(f"{word}: {similarity:.4f}")`

通过这个示例，您可以了解如何使用 Word2Vec 将词语转换为向量，并进行相似词语的查找。这种技术在自然语言处理任务中非常有用，例如文本分类、聚类、推荐系统等。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwMzg4NTQxMzhdfQ==
-->