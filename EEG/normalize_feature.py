import cPickle
import numpy as np

def normalize(res_data):
    print res_data[0, :, :, 1]
    maxs = np.max(np.abs(res_data), axis = 0)
    print maxs
    for y in range(9):
        for x in range(9):
            for ch_idx in range(5):
                if maxs[x, y, ch_idx] == 0: continue
                res_data[:, x, y, ch_idx] /= maxs[x, y, ch_idx]
    return res_data

if __name__ == "__main__":
    pre_batch = '/data/klng/git/EvolutionaryDNN/Datasets/EEG_data/'
    with open(pre_batch + 'graph_de.dat', 'rb') as f:
        origin_feature = cPickle.load(f)
    normalize(origin_feature)
    # with open(pre_batch + 'graph_de.dat', 'wb') as f:
    #     cPickle.dump(normalize(origin_feature), f)
