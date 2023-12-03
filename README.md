# 项目目录

TextModel

​      |____ NewsModel.model          sentencepiece在所有文本中训练的分词器模型

​      |____NewsModel.vocab            sentencepiece在所有文本中训练的分词器词表

model.py                                           模型文件，关于定义模型的所有代码

tokenizer.py                                      分词器，定义自然语言映射的decoder和encoder

train.py                                              训练模型脚本文件

NewsDataLoader.py                       加载新闻数据的批次加载器

# 训练数据集介绍

THUCTC中文文本分类数据集是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。采用了清华NLP组提供的THUCNews新闻文本分类数据集的子集，从[THUCNews](https://link.zhihu.com/?target=http%3A//thuctc.thunlp.org/)数据集中抽取了20万条新闻标题，**文本长度在20到30之间，少数文本有上千字**，一共**14个类别**。

token数量6亿+。

单卡2080Ti数据集语料库模型训练时间：1564秒

