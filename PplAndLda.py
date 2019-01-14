#!/usr/bin/python
# -*- coding: utf-8 -*-

import jieba.posseg as pseg
from gensim import corpora, models
import gensim
import pandas as pd

class PplAndLda:
    def __init__(self, labelList, ldaDimension, passes):
        # 名词
        # n 名词, nr 人名, nr1 汉语姓氏, nr2 汉语名字, nrj 日语人名, nrf 音译人名, ns 地名,
        # nsf 音译地名, nt 机构团体名, nz 其它专名, nl 名词性惯用语,  ng 名词性语素
        self.nounType = ['n', 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nl', 'ng']

        # 停用词
        self.stopWords = []
        for word in open('stop_words_ch.txt',encoding = 'GBK'):
            self.stopWords.append(word[:-1])

        # 标签字典
        self.labelList = labelList
        self.label2Type = {}
        for i, label in enumerate(self.labelList):
            self.label2Type[label] = i

        # LDA算法生成向量维度
        self.ldaDimension = ldaDimension

        # LDA模型遍历语料库次数
        self.passes = passes
        pass

    def ppl(self, contents):
        # 文本分词+去名词+去停用词
        results = []
        for content in contents:
            words = pseg.lcut(content, HMM=False)
            result = [word.word for word in words if word.flag in self.nounType
                      and not word.word in self.stopWords]
            results.append(result)
        return results
        pass

    def generateLdaModel(self, wordsList, modelDirPath = 'LdaModel'):
        # 生成词典
        dictionary = corpora.Dictionary(wordsList)
        dictionary.save(modelDirPath + '/dictionary.dict')

        # 生成LDA模型
        corpuses = [dictionary.doc2bow(text) for text in wordsList]
        ldaModel = gensim.models.ldamodel.LdaModel(corpuses, num_topics=self.ldaDimension,
                                                   id2word=dictionary, passes=self.passes)
        ldaModel.save(modelDirPath + '/LdaModel')

        return ldaModel
        pass

    def generateVectorData(self, wordsLists, labels, modelDirPath = 'LdaModel', tarDataPath = 'tmp/dataSet.csv'):
        # 载入字典与LDA模型
        ldaModel = gensim.models.ldamodel.LdaModel.load(modelDirPath + '/LdaModel')
        dictionary = corpora.Dictionary.load(modelDirPath + '/dictionary.dict')

        # 计算词袋向量
        corpuses = [dictionary.doc2bow(words) for words in wordsLists]

        # 计算各文本属于各LDA主题概率并生成DataFrame
        results = []
        for corpus in corpuses:
            tmp = ldaModel[corpus]
            result = [0 for i in range(100)]
            for i in tmp:
                result[i[0]] = i[1]
            results.append(result)
        headers = ['topic' + str(i) for i in range(self.ldaDimension)]
        print('results:', results)
        print('headers:', headers)
        data = pd.DataFrame(results, columns=headers)

        # 生成数字标签DataFrame
        labels = pd.DataFrame(labels, columns=['label'])
        labels['label'] = labels['label'].map(self.label2Type)

        print('writing csv data ...')
        # 合并数据和标签并存储csv文件
        data = data.join(labels)
        data.to_csv(tarDataPath)

        data = data.values
        data = data.astype(int)
        return data
        pass


if __name__ == '__main__':


    pass
