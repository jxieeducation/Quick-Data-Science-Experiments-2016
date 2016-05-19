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
