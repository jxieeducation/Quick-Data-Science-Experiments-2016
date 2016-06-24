'''
spark-submit --conf spark.driver.maxResultSize=1g --conf spark.yarn.executor.memoryOverhead=800 --num-executors 6 --executor-cores 1 --driver-memory 500m  --executor-memory 500m matrix_factorization.py 

https://spark.apache.org/docs/1.5.0/api/python/_modules/pyspark/mllib/recommendation.html
'''
import numpy as np
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

sc = SparkContext()

rating_data = sc.textFile("../ml-100k/u.data")
ratings = rating_data.map(lambda line: line.split("\t")).map(lambda data: Rating(int(data[0]), int(data[1]), float(data[2])))
# print rawRatings.first()

rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations, lambda_=0.01)
# model = ALS.trainImplicit(ratings, rank, numIterations, alpha=0.01)
# print model.productFeatures().collect()
print model.userFeatures().take(1)[0]

testdata = ratings.map(lambda p: (p[0], p[1]))
# t= testdata.first()
# print dir(t)

predictions = model.predictAll(testdata)
# print predictions.take(10)

predictionTuple = predictions.map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictionTuple)
# print predictionTuple.take(10)

MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
# print("Mean Squared Error = " + str(MSE))

# model.save(sc, "/tmp/myCollaborativeFilter")
# sameModel = MatrixFactorizationModel.load(sc, "/tmp/myCollaborativeFilter")

''' generating recommendations '''
# 1 
# print model.recommendProducts(789, 10)
# 2 
items = rating_data.map(lambda line: int(line.split("\t")[1])).distinct()
user_719_item_pairs = items.map(lambda item: (719, item))
# user_results = model.predictAll(user_719_item_pairs)
# print user_results.map(lambda rating: (rating[1], rating[2])).sortBy(lambda a: a[1]).take(10)





