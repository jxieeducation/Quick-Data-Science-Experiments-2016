from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Initializators, Mutators

def eval_func(chromosome):
	score = 0.0
	for value in chromosome:
		score += 0.8 - value
	return score

genome = G1DList.G1DList(1)
genome.evaluator.set(eval_func)
genome.setParams(rangemin=0.2, rangemax=0.8)
genome.initializator.set(Initializators.G1DListInitializatorReal)


ga = GSimpleGA.GSimpleGA(genome)
ga.setPopulationSize(100)
ga.setGenerations(120)
ga.setMutationRate(0.05)
ga.setCrossoverRate(0.85)

ga.evolve(freq_stats=10)
print ga.bestIndividual()
