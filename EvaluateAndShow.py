#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

class EvaluateAndShow:
    def __init__(self):
        # f-score的贝塔权重
        self.B = 1

        pass

    def Evaluate(self, predictLabel, actualLabel, labelList = list(range(10))):
        # 标签列
        labelAr = np.array(labelList).reshape([len(labelList), 1])

        # 预测各分类数据量
        predictLabelDataNum = np.sum(predictLabel == labelAr, axis = 1)

        # 真实各分类数据量
        actualLabelDataNum = np.sum(actualLabel == labelAr, axis = 1)

        # 正确各分类数据量
        correctLabelDataNum = np.sum(predictLabel[predictLabel == actualLabel] == labelAr, axis = 1)

        # 数据总量
        dataNum = len(predictLabel)

        # 正确分类数据量
        correctDataNum = np.sum(predictLabel == actualLabel)

        # 各分类正确率
        accuracyLabel = correctLabelDataNum / predictLabelDataNum
        print('各分类正确率:', accuracyLabel)

        # 平均正确率
        avgAccuracy = np.mean(accuracyLabel)
        print('平均正确率:', avgAccuracy)

        # 总体正确率
        allAccuracy = correctDataNum / dataNum
        print('总体正确率:', allAccuracy)

        # 各分类召回率
        recallLabel = correctLabelDataNum / actualLabelDataNum
        print('各分类召回率:', recallLabel)

        # 平均召回率
        avgRecall = np.mean(recallLabel)
        print('平均召回率:', avgRecall)

        # 各分类f-score
        f_scoreLabel = (1 + self.B * self.B) * (accuracyLabel * recallLabel) \
                  / (self.B * self.B * accuracyLabel + recallLabel)
        print('各分类f-score:', f_scoreLabel)

        pass


if __name__ == '__main__':
    pass