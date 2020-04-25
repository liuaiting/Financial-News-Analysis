# Financial News Analysis
[财经新闻分析](https://www.nowcoder.com/activity/2018cmbchina/bigdata/2)
## Table of Contents
- [Background](#Background)
- [Data](#Data)
- [Setup](#Setup)
- [Requirements](#Requirements)
- [Results](#Results)
- [Models](#Models)
    - [bm25](#bm25)
    - [average-word2vec](#average-word2vec)
    - [sequence-overlap](#sequence-overlap)
    - [LSI](#LSI)
    - [simhash](#simhash)


## Background
* 财经新闻作为重要却海量的投资数据，无时无刻不在影响着投资者们的投资决策，为了更好地提示客户当下新闻事件对应的投资机会和投资风险，本课以研发“历史事件连连看”为目的，旨在根据当前新闻内容从历史事件中搜索出相似新闻报道，后期可以结合事件与行情，辅助客户采取相应投资策略。
* 该赛题是让参赛者为每一条测试集数据寻找其最相似的TOP 20条新闻（不包含测试新闻本身），我们会根据参赛者提交的结果和实际的数据进行对比，采用mAP值作为评价指标。
评价指标
    该赛题是让参赛者为每一条测试集数据寻找其最相似的TOP 20条新闻，我们会根据参赛者提交的结果和实际的数据进行对比，采用mAP值作为评价指标，评分公式如下：
<center style="padding: 40px"><img width="70%" src=https://uploadfiles.nowcoder.com/images/20180423/59_1524486528472_5AF1D396F1D952E2B82E2EC949D0B8C7 /></center>
其中D表示测试集中新闻的总数量，Yd表示新闻d的n条真实相似新闻集合，无序，Yd={Yd1,Yd2,Yd3,……,Ydn }；Zd表示选手提交m条(赛题中m=20）相似新闻的有序集合Zd={Zd1,Zd2,Zd3,……,Zdm }；Zd中各元素的Rank值分别1,2,3,……,m,记为ri。对于集合K，|K|表示K中元素的个数,即|Yd |=n。

## Data

|训练集数据|[train_data.csv](https://static.nowcoder.com/activity/2018cmb/2/train_data.csv)|
|:--------- |-------------:|
|id |   训练集的新闻编号     |
|title| 训练集新闻（标题）  |

|测试集数据  |[test_data.csv](https://static.nowcoder.com/activity/2018cmb/2/test_data.csv)|
|:-------------- |--------------:|
|id    |   测试集的新闻编号     |
|title| 测试集新闻（标题）  |

## Setup
* Python 3.6

## Requirements

```bash
pip install -r requirements.txt
```

## Results

|Algorithm | mAP |
|:------------|------------:|
|sequence-overlap| 0.0587|
|average-word2vec| 0.0770|
|LSI(num_topics=1000)| 0.0854|
|LSI(num_topics=2000) | 0.0870|
|simhash | 0.0310|
|bm25(jieba)|___0.1137___|
|bm25(jieba&去停用词)| 0.1058|
|bm25(char) | 0.0850|
|bm25(thulac) |0.0932 |
|bm25(NLPIR) | 0.0957|

## Models
### bm25
##### bm25_model.py ：该模型效果最好，只给出该模型的代码
##### 把train_data.csv及test_data.csv放在path_corpus文件下，运行bm25_model.py即可
* 建立词袋模型
* 用gensim建立BM25模型
* 根据gensim源码，计算平均逆文档频率
* 利用BM25模型计算所有文本与搜索词的相关性（使用[gensim](https://radimrehurek.com/gensim/summarization/bm25.html)库）
* 找到最相关的top20文本

* 通过调整k1和b这两个参数，可以达到更好的效果

|bm25 | mAP |
|:------------|------------:|
|k1=1.5, b=0.75 | 0.1137|
|k1=1.5,b=0.85| 0.1182|
|k1=1, b=1| 0.12|
|k1=1.2, b=0.9| |
|k1=1.4, b=0.85| 0.1185|

### average-word2vec
1. 使用[jieba分词](https://github.com/fxsjy/jieba)对训练集数据进行分词
2. 使用google提供的[word2vec](https://code.google.com/archive/p/word2vec/)对分词后的训练语料进行训练，得到词向量（命令行参数设置为：./word2vec -train ../path_corpus/corpus_train.txt -output ../path_corpus/vec.txt -cbow 1 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -iter 10 -binary 0 -min-count 0 -save-vocab ../path_corpus/vocab.txt
3. 对于分词后的训练语料中的每一个样本，将句子中每一个词对应的词向量按位相加取平均值作为句子的向量表示
4. 采用[cosine similarity](http://scikit-learn.org/dev/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)作为文本相似度的度量标准，并取top20作为最终结果


### sequence-overlap
1. 该模型不需要对训练语料进行分词
2. 文本相似度采用[字符串的重叠度](https://docs.python.org/2/library/difflib.html#sequencematcher-objects)来度量

### LSI
1. 分词、去停用词
2. 词袋模型向量化文本
3. TF-IDF模型向量化文本
4. LSI模型向量化文本（使用[gensim](https://github.com/RaRe-Technologies/gensim)库）
5. 计算相似度

### simhash

<center style="padding: 40px"><img width="70%" src=http://7viirv.com1.z0.glb.clouddn.com/simhash.jpg /></center>

1. 过滤清洗，提取n个特征关键词
2. 特征加权，tf-idf
3. 对关键词进行hash降维01组成的签名（上述是6位）
4. 然后向量加权，对于每一个6位的签名的每一位，如果是1，hash和权重正相乘，如果为0，则hash和权重负相乘，至此就能得到每个特征值的向量。
5. 合并所有的特征向量相加，得到一个最终的向量，然后降维，对于最终的向量的每一位如果大于0则为1，否则为0，这样就能得到最终的simhash的指纹签名（使用[simhash](https://github.com/leonsim/simhash)库）










