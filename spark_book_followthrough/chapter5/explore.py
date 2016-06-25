'''
spark-submit --num-executors 6 --executor-cores 1 explore.py 

https://www.kaggle.com/c/stumbleupon/data
'''

import numpy as np
from pyspark import SparkContext
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext()

rawData = sc.textFile("train_noheader.tsv")
records = rawData.map(lambda line: line.split("\t"))
# print records.first()

def formatRecord(record):
	label = 0 if record[-1] == '"0"' else 1
	features = record[4:-1]
	features = [0.0 if elem[1:-1] == "?" else float(elem[1:-1]) for elem in features]
	return LabeledPoint(label, features)
data = records.map(formatRecord)
data.cache()
# print data.take(10)

def formatRecordNonzero(record):
	label = 0 if record[-1] == '"0"' else 1
	features = record[4:-1]
	features = [0.0 if elem[1:-1] == "?" or float(elem[1:-1]) < 0 else float(elem[1:-1]) for elem in features]
	return LabeledPoint(label, features)
dataNonzero = records.map(formatRecordNonzero)
# print dataNonzero.take(10)

from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD
# from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.tree import DecisionTree

numIterations = 10
lrModel = LogisticRegressionWithSGD.train(data, numIterations)
svmModel = SVMWithSGD.train(data, numIterations)
# nbModel = NaiveBayes.train(dataNonzero, numIterations)
# dtModel = DecisionTree.trainClassifier(data, numClasses=2, impurity='gini', maxDepth=5, maxBins=32)

# print data.first()
# print lrModel.predict(data.first().features)
predictions = lrModel.predict(data.map(lambda labeledPoint: labeledPoint.features))
print predictions.take(5)


