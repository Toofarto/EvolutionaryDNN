from tf import *
import numpy as np
import itertools

def subtree_distance(weight, index_1, index_2):
    return np.sum(np.square(weight[index_1, :] - weight[index_2, :]))

def find_merge(weight, other_weight):
    best_index_1, best_index_2, shortest_distance = 0, 1, 0
    for (i, j) in itertools.combinations(range(len(weight)), 2):
        distance = subtree_distance(weight, i, j)
        if display < shortest_distance:
            best_index_1, best_index_2, shortest_distance = i, j, distance
    weight[best_index_1, :] = (weight[best_index_2, :] + weight[best_index_2, :])/2

def crossover(weight1, weight2, n_crossover):
    sz_w1, sz_w2 = np.shape(weight1), np.shape(weight2)
    assert(sz_w1 == sz_w2)
    cross_set = np.array([np.random.choice(np.arange(sz_w1[0]), n_crossover) for i in range(2)])
    for i in range(n_crossover):
        index1, index2 = cross_set[:, i]
        weight1[index1, :], weight2[index2, :] = weight2[index2, :], weight1[index1, :]
