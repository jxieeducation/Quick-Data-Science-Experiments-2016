'''
spark-submit --conf spark.driver.maxResultSize=1g --conf spark.yarn.executor.memoryOverhead=800 --num-executors 6 --executor-cores 1 --driver-memory 500m  --executor-memory 500m exploration.py 
'''
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import datetime

from pyspark import SparkContext
sc = SparkContext()

user_data = sc.textFile("../ml-100k/u.user")
# print user_data.first() # "1|24|M|technician|85711"

user_ages = user_data.map(lambda line: int(line.split("|")[1]))
# print user_ages.take(10) # [24, 53, 23, 24, 33, 42, 57, 36, 29, 53]
# plt.hist(user_ages.collect(), bins=20, color="lightblue", normed=True)
# plt.show()

user_occupations = user_data.map(lambda line: line.split("|")[3])
# print user_occupations.take(3)
occupation_counts = user_occupations.map(lambda occ: (occ, 1)).reduceByKey(lambda x,y: x + y)
# x_axis1 = np.array([c[0] for c in occupation_counts.collect()])
# y_axis1 = np.array([c[1] for c in occupation_counts.collect()])
# pos = np.arange(len(x_axis1))
# ax = plt.axes()
# ax.set_xticklabels(x_axis1)
# plt.bar(pos, y_axis1, 1.0)
# plt.show()

movie_data = sc.textFile("../ml-100k/u.item")
def convert_year(x):
	try:
		return int(x[-4:])
	except:
		return 1990
movie_years = movie_data.map(lambda line: line.split("|")[2]).map(convert_year)
movie_ages = movie_years.map(lambda year: 1998 - year).countByValue()
ages = movie_ages.values()
bins = movie_ages.keys()


rating_data = sc.textFile("../ml-100k/u.data")
ratings = rating_data.map(lambda line: int(line.split("\t")[2])).countByValue()
ratings_vals = ratings.values()
ratings_keys = ratings.keys()

num_ratings_per_user = rating_data.map(lambda line: line.split("\t")[0]).countByValue()
rating_freq = num_ratings_per_user.values()
# plt.hist(rating_freq, bins=100)
# plt.show()


''' 1 of k encoding '''
distinct_occupations = user_occupations.distinct()
occupationLUT = {}
for occupation in distinct_occupations.collect():
	occupationLUT[occupation] = len(occupationLUT)
def getOccupationEncoding(occupation, LUT):
	binary_x = np.zeros(len(LUT))
	binary_x[LUT[occupation]] = 1
	return binary_x
# print getOccupationEncoding("administrator", occupationLUT)

rating_times = rating_data.map(lambda line: datetime.datetime.fromtimestamp(int(line.split("\t")[-1])))
rating_hours = rating_times.map(lambda time: time.hour)
# print rating_hours.take(10)

x = np.random.rand(10)
from pyspark.mllib.feature import Normalizer
normalizer = Normalizer() # scales every row by itself
vector = sc.parallelize([x, x + 1, x + 2])
# print normalizer.transform(vector).collect() 

