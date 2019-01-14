#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pandas as pd

class LoadData:
    def __init__(self):
        pass

    def loadScData(self, dataPath = '../THUCNews'):
        texts = []
        labels = []

        for dirPath, _, fileList in os.walk(dataPath):
            print('Reading ' + dirPath + '...')
            # 获取标签
            label = dirPath.split('/')[-1]

            for fileName in fileList:
                # 读取文本
                filePath = dirPath + '/' + fileName
                with open(filePath, encoding='utf-8') as file:
                    texts.append(file.read())

                # 添加标签
                labels.append(label)

        return texts, labels
        pass

    def loadCsvData(self, filePath):
        data = pd.read_csv(filePath)

        return data.values
        pass

if __name__ == '__main__':
    ld = LoadData()
    X, Y = ld.loadScData()
    for i in range(len(X)):
        print(X[i], Y[i])
    pass
