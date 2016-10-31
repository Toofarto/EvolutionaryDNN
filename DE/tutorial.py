from PyGMO import *
prob = problem.schewfel(dem = 50)
algo = algorithm.de(gen = 500)
isl = island(algo, prob, 20)
print isl.population.champion.f
isl.evolve(10)
print isl.population.champion.f
