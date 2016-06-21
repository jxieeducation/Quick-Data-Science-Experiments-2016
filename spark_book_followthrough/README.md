spark-submit --conf spark.driver.maxResultSize=1g --conf spark.yarn.executor.memoryOverhead=800 --num-executors 6 --executor-cores 1 --driver-memory 500m  --executor-memory 500m spark_som.py 

wget http://files.grouplens.org/datasets/movielens/ml-100k.zip

IPYTHON=1 IPYTHON_OPTS="--pylab" pyspark

