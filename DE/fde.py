import pyfde

def fitness(p):
    learning_rate y = p[0], p[1]
    res = CNN_model(learning_rate)
    return res

solver = pyfde.ClassicDE(fitness, n_dim=2, 
        n_pop=40, limits=(1e-6, 1e2))
solver.cr, solver.f = 0.9, 0.45
best, fit = solver.run(n_it=150)


