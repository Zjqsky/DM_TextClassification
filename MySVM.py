#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
import numpy as np
import pandas as pd
import pickle

class MySVM:
   def __init__(self, typeNum):
       self.typeNum = typeNum
       self.svmModel = None
       pass

   def train(self, data, labels, modelDirPath = 'SVMModel'):
       # SVM模型建立
       # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
       # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
       self.svmModel = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
       self.svmModel.fit(data, labels.ravel())

       self.saveModel(self.svmModel, modelDirPath)
       pass

   def saveModel(self, model, modelDirPath = 'SVMModel'):
       with open(modelDirPath + '/svmModel.pickle', 'wb') as modelFile:
           pickle.dump(model, modelFile)
       pass

   def loadModel(self, modelDirPath = 'SVMModel'):
       with open(modelDirPath + '/svmModel.pickle', 'rb') as modelFile:
           self.svmModel = pickle.load(modelFile)
       pass

   def predict(self, data, modelDirPath = 'SVMModel'):
       if self.svmModel == None:
           self.loadModel(modelDirPath)

       return self.svmModel.predict(data)
       pass


if __name__ == '__main__':
    pass