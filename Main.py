#!/usr/bin/python
# -*- coding: utf-8 -*-

import LoadData as Ld
import PplAndLda as Ppl
import MySVM as Ms
import MyNaiveBayes as Mnb
import EvaluateAndShow as Eas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import json
import TfIdf1

class Run:
    def __init__(self):
        print('Initiating ...')
        # 标签
        self.labelList = ['财经', '房产', '股票', '教育', '科技',
                          '社会', '时政', '体育', '游戏', '娱乐']

        self.labelLen = len(self.labelList)

        # 算法生成向量维度
        self.dimension = 5000

        # LDA模型遍历语料库次数
        self.ldaPasses = 10

        #print('Loading data ...')
        # 导入数据
        #self.dL = Ld.LoadData()
        #self.trainData = self.dL.loadCsvData('tmp/trainDataSet.csv')
        #self.testData = self.dL.loadCsvData('tmp/testDataSet.csv')
        
        #self.texts, self.labels  = self.dL.loadScData('../THUCNews_final')
        #print('datalen:',len(self.texts))

        #print('Spliting data ...')
        #self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.texts, self.labels, test_size=0.5)

        # SVM对象
        self.svm = Ms.MySVM(self.labelLen)

        # 贝叶斯对象
        self.valuesNumForBayes = 50
        self.naiveBayes = Mnb.MyNaiveBayes(self.labelLen, self.valuesNumForBayes)

        # EvaluateAndShow对象
        self.eas  = Eas.EvaluateAndShow()

        pass

    def foreProcessOfTrainByLda(self, texts, labels, tarDataPath = 'tmp/trainDataSet.csv'):
        # 文本分词+去名词+去停用词
        operateFunction = Ppl.PplAndLda(self.labelList, self.dimension, self.ldaPasses)
        resultOfPpl = operateFunction.ppl(texts)
        # 生成字典和Lda模型
        operateFunction.generateLdaModel(resultOfPpl)
        # 使用模型生成预处理数据集并存入csv文件
        data = operateFunction.generateVectorData(resultOfPpl, labels, tarDataPath = tarDataPath)
        return data

    def foreProcessOfTestByLda(self, texts, labels, tarDataPath = 'tmp/testDataSet.csv'):
        # 文本分词+去名词+去停用词
        operateFunction = Ppl.PplAndLda(self.labelList, self.dimension, self.ldaPasses)
        # resultOfPpl = operateFunction.ppl(texts)
        # 使用模型生成预处理数据集并存入csv文件
        data = operateFunction.generateVectorData(texts, labels, tarDataPath = tarDataPath)
        return data


    def trainByBayes(self, data):
        X = data[:, :-1]
        Y = data[:, -1]
        Y = Y.astype('int')

        #
        print('prepro...')
        X = X / np.max(X, axis = 1).reshape(X.shape[0], 1)
        X = X * self.valuesNumForBayes + 1
        X = np.minimum(X, self.valuesNumForBayes)
        X = np.rint(X)
        X = X.astype('int')

        print('training...')
        t = time.time()
        self.naiveBayes.train(X, Y)
        t = time.time() - t
        print('train Bayes using time:', t)

        t = time.time()
        predictResult = self.naiveBayes.predict(X)
        t = time.time() -t
        print('predict Bayes on trainSet using time:', t)	

        self.eas.Evaluate(predictResult, Y, list(range(self.labelLen)))

        pass

    def trainBySvm(self, data):
        X = data[:, :-1]
        Y = data[:, -1]

        t = time.time()
        self.svm.train(X, Y)
        t = time.time() - t
        print('train SVM using time:', t)

        t = time.time()
        predictResult = self.svm.predict(X)
        t = time.time() - t
        print('predict SVM on trainSet using time:', t)

        self.eas.Evaluate(predictResult, Y, list(range(self.labelLen)))

        pass

    def testByBayes(self, data):
        X = data[:, :-1]
        Y = data[:, -1]
        Y = Y.astype('int')

        #
        print('prepro...')
        X = X / np.max(X, axis = 1).reshape(X.shape[0], 1)
        X = X * self.valuesNumForBayes + 1
        X = np.minimum(X, self.valuesNumForBayes)
        X = np.rint(X)
        X.astype('int')

        t = time.time()
        predictResult = self.naiveBayes.predict(X)
        t = time.time() - t
        print('predict Bayes on testSet using time:', t)

        self.eas.Evaluate(predictResult, Y, list(range(self.labelLen)))

        pass

    def testBySvm(self, data):
        X = data[:, :-1]
        Y = data[:, -1]

        t = time.time()
        predictResult = self.svm.predict(X)
        t = time.time() - t
        print('predict SVM on testSet using time:', t)

        self.eas.Evaluate(predictResult, Y, list(range(self.labelLen)))

        pass

    def process1(self):
        print('Loading data ...')
        self.texts, self.labels  = self.dL.loadScData('../THUCNews_final')
        print('datalen:',len(self.texts))

        print('Spliting data ...')
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.texts, self.labels, test_size=0.5)

        print('foreProcessOfTrain ...')
        trainData = self.foreProcessOfTest(self.x_train, self.y_train,'tmp/trainDataSet.csv')

        print('foreProcessOfTest ...')
        testData = self.foreProcessOfTest(self.x_test, self.y_test,'tmp/testDataSet.csv')

        print('trainByBayes ...')
        # self.trainByBayes(trainData)

        print('testByBayes ...')
        # self.testByBayes(testData)

        print('trainBySvm ...')
        # self.trainBySvm(trainData)

        print('testBySvm ...')
        # self.testBySvm(testData)

        pass




    def dataProcessByPpl(self):
        print('Loading data ...')
        self.texts, self.labels  = self.dL.loadScData('../THUCNews_final')
        print('datalen:',len(self.texts))

        print('Spliting data ...')
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.texts, self.labels, test_size=0.5)

        operateFunction = Ppl.PplAndLda(self.labelList, self.dimension, self.ldaPasses)
        print('Ppling trainData....')
        resultOfPpl = operateFunction.ppl(self.x_train)
        resultOfPpl.append(self.y_train)
        print('save trainPplData...')
        with open('tmp/trainWordsList', 'w') as f:		
            json.dump(resultOfPpl,f)
        print('Ppling testData....')
        resultOfPpl = operateFunction.ppl(self.x_test)
        resultOfPpl.append(self.y_test)
        print('save testPplData...')
        with open('tmp/testWordsList', 'w') as f:
            json.dump(resultOfPpl,f)
        pass

    def dataProcessByTfIdf(self):
        print('loading train data ...')
        #data = []
        #with open('tmp/trainWordsList') as f:
        #    data = json.load(f)
        #self.x_train = data[:-1]
        #self.y_train = data[-1]

        operateFunction = TfIdf1.TFIDF(self.labelList, self.dimension)
        #print('TF-IDF run ...')
        #operateFunction.generateTfIdf(self.x_train, self.y_train)

        print('loading test data ...')        
        with open('tmp/testWordsList') as f:
            data = json.load(f)
        self.x_test = data[:-1]
        self.y_test = data[-1]

        print('using TF-IDF...')
        data = operateFunction.useTfidf(self.x_test, self.y_test)
        pass

    def dataProcessByLda(self):
        data = []
        with open('tmp/trainWordsList') as f:
            data = json.load(f)
        self.x_train = data[:-1]
        self.y_train = data[-1]
        self.foreProcessOfTest(self.x_train, self.y_train,'tmp/trainDataSet.csv')

        
        with open('tmp/testWordsList') as f:
            data = json.load(f)
        self.x_test = data[:-1]
        self.y_test = data[-1]

        self.foreProcessOfTest(self.x_test, self.y_test,'tmp/testDataSet.csv')
        pass

    def processTrainBayes(self):
        print('Loading data ...')
        # 导入数据
        self.dL = Ld.LoadData()
        self.trainData = self.dL.loadCsvData('tmp/trainDataSetByT.csv')
        #self.testData = self.dL.loadCsvData('tmp/testDataSetByT.csv')

        print('training ...')
        self.trainByBayes(self.trainData)

    def processTestBayes(self):
        print('Loading data ...')
        # 导入数据
        self.dL = Ld.LoadData()
        #self.trainData = self.dL.loadCsvData('tmp/trainDataSetByT.csv')
        self.testData = self.dL.loadCsvData('tmp/testDataSetByT.csv')

        print('testing ...')
        self.testByBayes(self.testData)

    def processTrainSVM(self):
        print('Loading data ...')
        # 导入数据
        self.dL = Ld.LoadData()
        self.trainData = self.dL.loadCsvData('tmp/trainDataSetByT.csv')
        #self.testData = self.dL.loadCsvData('tmp/testDataSetByT.csv')

        print('training ...')
        self.trainBySvm(self.trainData)

    def processTestSVM(self):
        print('Loading data ...')
        # 导入数据
        self.dL = Ld.LoadData()
        #self.trainData = self.dL.loadCsvData('tmp/trainDataSetByT.csv')
        self.testData = self.dL.loadCsvData('tmp/testDataSetByT.csv')

        print('testing ...')
        self.testBySvm(self.testData)

if __name__ == '__main__':
    #Run().dataProcessByPpl()
    #Run().dataProcessByLda()
    Run().dataProcessByTfIdf()
    #Run().processTrainBayes()
    #Run().processTestBayes()
    #Run().processTrainSVM()
    #Run().processTestSVM()
    pass
