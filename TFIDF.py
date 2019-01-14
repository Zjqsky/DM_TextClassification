
# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import operator
import numpy as np
import pandas as pd
import jieba.posseg as pseg
import pickle

class TFIDF:
    def __init__(self, labelList, tiDimension):
        # 名词
        # n 名词, nr 人名, nr1 汉语姓氏, nr2 汉语名字, nrj 日语人名, nrf 音译人名, ns 地名,
        # nsf 音译地名, nt 机构团体名, nz 其它专名, nl 名词性惯用语,  ng 名词性语素
        self.nounType = ['n', 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nl', 'ng']

        # 停用词
        self.stopWords = []
        for word in open('stop_words_ch.txt', encoding='GBK'):
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
            words = pseg.lcut(content, HMM=False)
            result = [word.word for word in words if word.flag in self.nounType
                      and not word.word in self.stopWords]
            results.append(result)

        return results
        pass

    def TfIdfFeature(self, list_words, modelDirPath = 'TFIDFModel'):
        doc_frequency = defaultdict(int)
        for word_List in list_words:
            for i in word_List:
                doc_frequency[i] += 1

        word_tf = {}
        for i in doc_frequency:
            word_tf[i] = doc_frequency[i]/sum(doc_frequency.values())

        doc_num = len(list_words)
        word_idf = {}
        word_doc = defaultdict(int)
        for i in doc_frequency:
            for j in list_words:
                if i in j:
                    word_doc[i] += 1
        for i in doc_frequency:
            word_idf[i] = math.log((doc_num + 1)/(word_doc[i] + 1)) + 1

        word_tf_idf = {}
        for i in doc_frequency:
            word_tf_idf[i] = word_tf[i] * word_idf[i]

        dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
        dict_feature_select = dict_feature_select[:self.tiDimension]

        with open(modelDirPath + '/tfidfModel.pickle', 'wb') as modelFile:
            pickle.dump(dict_feature_select, modelFile)

        return dict_feature_select

    def tfIdfUsage(self, list_words, labels, modelDirPath = 'TFIDFModel', tarDataPath = 'tmp/dataSet.csv'):
        frequencyInDocs = []
        for result in list_words:
            resultDict = {}
            for word in result:
                if not word in resultDict:
                    resultDict[word] = 0
                resultDict[word] += 1
            frequencyInDocs.append(resultDict)

        dimension = self.tiDimension
        result = np.zeros([len(frequencyInDocs), dimension])

        dict_feature_select = None
        with open(modelDirPath + '/tfidfModel.pickle', 'rb') as modelFile:
            dict_feature_select = pickle.load(modelFile)

        indexDict = {}
        for index, key in enumerate(dict_feature_select):
            indexDict[key] = index

        for i, frequencyInDoc in enumerate(frequencyInDocs):
            for frequency in frequencyInDoc:
                if frequency in dict_feature_select:
                    result[i, indexDict[frequency]] = frequencyInDoc[frequency]

        data = pd.DataFrame(result)

        # 生成数字标签DataFrame
        labels = pd.DataFrame(labels, columns=['label'])
        labels['label'] = labels['label'].map(self.label2Type)

        data = data.join(labels)
        data.to_csv(tarDataPath)

        data = data.values
        return data

        pass

    if __name__ == '__main__':
        pass
