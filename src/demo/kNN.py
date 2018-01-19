'''
Created on 2018年1月14日

@author: IL MARE
'''
import numpy as np
import os

print("import {0}".format(__name__))
def createDatSet():#测试数据集
    group = np.array([[1.0,1.1], [1.0,1.0], [0, 0], [0, 0.1]], dtype=np.float)
    labels = ['A', 'A', 'B','B']
    return group, labels

def classify0(intX:"需要被分类的数据", dataSet:"数据集", labels:"数据集标签", k:"k值")->tuple:
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(intX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistanceIndex = distance.argsort()
    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDistanceIndex[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    tmp_count = -1
    tmp_flag = "-1"
    for item in classCount.items():
        if tmp_count < item[1]:
            tmp_flag = item[0]
            tmp_count = item[1]
    return tmp_flag if tmp_flag != "-1" else None

def file2matrix(filename):#从文件中读取数据
    try:
        fp = open(filename, "r")
        arrayLine = fp.readlines()
        numberOfLine = len(arrayLine)
        returnMat = np.zeros((numberOfLine, 3))
        classLabelVector = []
        index = 0
        for line in arrayLine:
            line = line.strip()
            listFromLine = line.split("\t")
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector
    except Exception as e:
        print(e)
    finally:
        fp.close()
        
def autoNormal(dataSet):#正规化数据
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    rangeVal = maxVal - minVal
    normalDataSet = np.zeros(dataSet.shape, dtype = np.float)
    m = dataSet.shape[0]
    normalDataSet = dataSet - np.tile(minVal, (m, 1))
    normalDataSet /= np.tile(rangeVal, (m, 1))
    return normalDataSet, rangeVal, minVal

def datingClassTest():
    ratio = 0.3
    datingDataMat, datingLabel = file2matrix("./datingTestSet2.txt")
    datingDataMat, rangeVal, minVal = autoNormal(datingDataMat)
    m = datingDataMat.shape[0]
    numOfTestVec = int(m * ratio)
    error = 0
    for i in range(numOfTestVec):
        classifyRes = classify0(datingDataMat[i,:], datingDataMat[numOfTestVec:m, :], datingLabel[numOfTestVec:m], 3)
        print("the classifier came back with: {0}, the real answer is: {1}".format(classifyRes, datingLabel[i]))
        if classifyRes != datingLabel[i]:
            error += 1
    print("the total error rate is {0:.3f}".format(error / float(numOfTestVec)))

def img2Vector(filename):
    try:
        fp = open(filename, "r")
        returnVector = []
        for item in fp.readlines():
            item = item.rstrip()
            for num in item:
                returnVector.append(num)
        return np.array([returnVector], dtype = np.float)
    except Exception as e:
        print(e)
    finally:
        fp.close()
        
def handWritingTest():
    filePath = r"./trainingDigits"
    DirList = os.listdir(filePath)
    m = len(DirList)
    trainSet = np.zeros((m, 1024))
    labelSet = []
    for i in range(m):
        path = DirList[i]
        labelSet.append(path.split("_")[0])
        returnVect = img2Vector(filePath + "\\" + path)
        trainSet[i, :] = returnVect
    testPath = r".\testDigits"
    DirList = os.listdir(testPath)
    m = len(DirList)
    error = 0
    for i in range(m):
        path = DirList[i]
        label = path.split("_")[0]
        returnVect = img2Vector(testPath + "/" + path)
        tmp_res = classify0(returnVect[0, :], trainSet, labelSet, 10)
        print("the classifier came back with: {0}, the real answer is: {1}".format(tmp_res, label))
        if tmp_res != label:
            error += 1
    print("the total error ratio is {0:.5f}".format(error / m))
