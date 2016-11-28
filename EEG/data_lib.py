import numpy as np
import cPickle


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

class NP_Dataset(object):
    def __init__(self, pX, pY):
        self._X = pX
        #self._Y = [map(lambda x: x == y, range(17)) for y in pY]
        self._Y = pY
        assert np.shape(self._X)[0] == np.shape(self._Y)[0]
        self._n_sample = np.shape(self._X)[0]
        self._index_in_epoch = 0
        self._epoch_completed = 0
        self.shuffle()

    def shuffle(self):
        perm = np.arange(self._n_sample) 
        np.random.shuffle(perm)
        self._X = self._X[perm]
        self._Y = self._Y[perm]

    def next_batch(self, batch_size):
        if (batch_size == -1): return (self._X[:], self._Y[:])
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._n_sample:
            assert batch_size <= self._n_sample
            self._epoch_completed += 1
            # Shuffle
            self.shuffle()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._X[start:end], self._Y[start:end]

    def get_epoch(self):
        return self._epoch_completed

    def get_n_sample(self):
        return self._n_sample
