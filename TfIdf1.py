# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import operator
import numpy as np
import pandas as pd
import jieba.posseg as pseg
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDF:
    def __init__(self, labelList, tiDimension):
        # 名词
        # n 名词, nr 人名, nr1 汉语姓氏, nr2 汉语名字, nrj 日语人名, nrf 音译人名, ns 地名,
        # nsf 音译地名, nt 机构团体名, nz 其它专名, nl 名词性惯用语,  ng 名词性语素
        self.nounType = ['n', 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nl', 'ng']

        # 停用词
        self.stopWords = []
        for word in open('stop_words_ch.txt', encoding = 'GBK'):
            self.stopWords.append(word[:-1])

        # 标签字典
        self.labelList = labelList
        self.label2Type = {}
        for i, label in enumerate(self.labelList):
            self.label2Type[label] = i

        # TFIDF算法生成向量维度
        self.tiDimension = tiDimension
        pass

    def ppl(self, list_words):
        # 文本分词+去名词+去停用词
        results = []
        for content in list_words:
            words = pseg.lcut(content, HMM=True)
            result = [word.word for word in words if word.flag in self.nounType
                      and not word.word in self.stopWords]
            results.append(result)

        return results
        pass

    def generateTfIdf(self, wordsList, labels, modelDirPath = 'TFIDFModel',
                      tarDataPath='tmp/trainDataSetByT.csv'):
        strList = []
        for words in wordsList:
            wordsStr = ' '.join(words)
            strList.append(wordsStr)

        # 调用包函数
        tfidf = TfidfVectorizer(analyzer='word', max_features=self.tiDimension, lowercase = False)
        dataSet = tfidf.fit_transform(strList)
        with open(modelDirPath + '/tfidfModel.pickle', 'wb') as modelFile:
            pickle.dump(tfidf, modelFile)

        dataSet = pd.DataFrame(dataSet.toarray())
        # 生成数字标签DataFrame
        labels = pd.DataFrame(labels, columns=['label'])
        labels['label'] = labels['label'].map(self.label2Type)

        data = dataSet.join(labels)
        data.to_csv(tarDataPath)

        data = data.values
        return data

    def useTfidf(self, wordsList, labels, modelDirPath = 'TFIDFModel',
                tarDataPath='tmp/testDataSetByT.csv'):
        strList = []
        for words in wordsList:
            wordsStr = ' '.join(words)
            strList.append(wordsStr)

        tfidf = None
        with open(modelDirPath + '/tfidfModel.pickle', 'rb') as modelFile:
            tfidf = pickle.load(modelFile)
        dataSet = tfidf.transform(strList)
        dataSet = pd.DataFrame(dataSet.toarray())

        # 生成数字标签DataFrame
        labels = pd.DataFrame(labels, columns=['label'])
        labels['label'] = labels['label'].map(self.label2Type)

        data = dataSet.join(labels)
        data.to_csv(tarDataPath)

        data = data.values
        return data
        pass

    if __name__ == '__main__':
        pass
