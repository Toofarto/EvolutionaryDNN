import cPickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

pre_batch = '/data/klng/git/EvolutionaryDNN/Datasets/EEG_data/'
feature = unpickle(pre_batch + 'graph_de_normalized.dat')
print feature[19]
