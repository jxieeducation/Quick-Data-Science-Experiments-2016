import pandas as pd
import xgboost as xgb

df = pd.DataFrame({'x':[1,2,3], 'y':[10,20,30]})
X_train = df.drop('y',axis=1)
Y_train = df['y']
T_train_xgb = xgb.DMatrix(X_train, Y_train)

params = {"objective": "reg:linear", "booster":"gblinear", "eta": 0.5}
watchlist = [(T_train_xgb,'train')]
res = {}
gbm = xgb.train(dtrain=T_train_xgb, params=params, 
	evals_result=res, evals=watchlist, num_boost_round=3)
fitness = res['train']['rmse'][-1]

print fitness

from pyevolve import G1DList
from pyevolve import GSimpleGA

def eval_func(chromosome):
   score = 0.0
   # iterate over the chromosome
   for value in chromosome:
      if value==0:
         score += 1
   return score
   # return random.random()

genome = G1DList.G1DList(20)
genome.evaluator.set(eval_func)
ga = GSimpleGA.GSimpleGA(genome)
ga.setGenerations(120000)
ga.evolve(freq_stats=10)
print ga.bestIndividual()

