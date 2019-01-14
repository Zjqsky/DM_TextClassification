#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import numpy as np

class MyNaiveBayes:
    def __init__(self, typeNum, valuesNum = 100):
        self.probabilityDict = None

        # 分类类别数
        self.typeNum = typeNum

        # 标签概率零处理参数
        self.L = 1

        # 数据值量
        self.valuesNum = valuesNum

        # 数据概率零处理参数
        self.D = 1

        self.labelDivider = None
        self.dataDivider = None

        pass

    def train(self, data, labels, modelDirPath = 'NaiveBayesModel'):
        # 概率字典
        self.probabilityDict = {}
        # 数据大小
        dataSize = len(labels)
        # 数据维度
        dataDimension = data.shape[1]

        # 获取各标签集合
        labelSetDict = {}
        for i, label in enumerate(labels):
            if not label in labelSetDict:
                labelSetDict[label] = set()
            labelSetDict[label].add(i)

        # 计算先验概率
        self.labelDivider = dataSize + self.L * self.typeNum
        for label in labelSetDict:
            key = 'label_' + str(label)
            self.probabilityDict[key] = (len(labelSetDict[label]) + self.L) / self.labelDivider

        # 获取各属性集集合,计算条件概率
        self.dataDivider = dataSize + self.valuesNum * self.D
        for i in range(dataDimension):
            dataSetDict = {}
            for j, n in enumerate(data[:, i]):
                if not n in dataSetDict:
                    dataSetDict[n] = set()
                dataSetDict[n].add(j)
            for label in labelSetDict:
                for n in dataSetDict:
                    resultKey = 'data_D' + str(i) + '_' + str(n) + '|' + 'label_' + str(label)
                    labelKey = 'label_' + str(label)
                    self.probabilityDict[resultKey] = (len(labelSetDict[label] & dataSetDict[n]) + self.D) \
                                                      / self.dataDivider / self.probabilityDict[labelKey]

        # 贝叶斯模型导出
        self.saveModel(self.probabilityDict, modelDirPath)

        pass

    def saveModel(self, probabilityDict, modelDirPath = 'NaiveBayesModel'):
        # 贝叶斯模型导出
        modelFile = open(modelDirPath + '/NaiveBayesModel.txt', 'w')
        json.dump(probabilityDict, modelFile)
        modelFile.close()
        pass

    def loadModel(self, modelDirPath = 'NaiveBayesModel'):
        # 贝叶斯模型导入
        modelFile = open(modelDirPath + '/NaiveBayesModel.txt')
        self.probabilityDict = json.load(modelFile)
        modelFile.close()

    def predict(self, data):
        self.loadModel()
        
        # 数据大小, 数据维度
        if len(data.shape) == 1:
            dataSize = 1
            dataDimension = data.shape[0]
        else:
            dataSize, dataDimension = data.shape

        # 预测
        probabilityArray = np.zeros([dataSize, self.typeNum])
        for t in range(self.typeNum):
            keyPri = 'label_' + str(t)
            for i in range(dataSize):
                probabilityArray[i, t] = self.probabilityDict[keyPri]
                for j in range(dataDimension):
                    keyCondition = 'data_D' + str(j) + '_' + str(data[i, j]) + '|' + 'label_' + str(t)
                    if  not keyCondition in self.probabilityDict:
                        probabilityArray[i, t] *= (self.D / self.dataDivider / self.probabilityDict[keyPri])
                    else:
                        probabilityArray[i, t] *= self.probabilityDict[keyCondition]
        predictResult = np.argmax(probabilityArray, axis=1)

        return predictResult
        pass


if __name__ == '__main__':
    pass
