from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Selectors
import pyevolve
import code

# The generation that Pyevolve will enter on
# the interactive mode
INTERACTIVE_STOP = 50

def evolve_callback(ga_engine):
   """ The callback function to enter on interactive mode"""
   generation = ga_engine.getCurrentGeneration()

   if generation == INTERACTIVE_STOP:
      from pyevolve import Interaction
      interact_banner = "## Pyevolve v.%s - Interactive Mode ##" \
                        % (pyevolve.__version__,)
      session_locals = { "ga_engine"  : ga_engine,
                         "population" : ga_engine.getPopulation(),
                         "pyevolve"   : pyevolve,
                         "it"         : Interaction}
      print
      code.interact(interact_banner, local=session_locals)
   return False

def eval_func(chromosome):
   """ The evaluation function """
   score = 0.0
   for value in chromosome:
      if value==0:
         score += 0.1
   return score

genome = G1DList.G1DList(30)
genome.setParams(rangemin=0, rangemax=10)
genome.evaluator.set(eval_func)
ga = GSimpleGA.GSimpleGA(genome)
ga.setGenerations(500)
ga.stepCallback.set(evolve_callback)
ga.evolve(freq_stats=10)
print ga.bestIndividual()
