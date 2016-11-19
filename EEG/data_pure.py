from data_lib import *


def to_x(data):
    sz_group = 10
    data.resize(len(data) * 40, 40, 8064)
    data = data.swapaxes(1, 2)
    # normalize
    max_feature = [max(abs(np.max(data[:, :, i])), abs(np.min(data[:, :, i]))) for i in range(40)]
    # the fist 32 channel is EE of the brain, regards as image
    max_feature = [np.max(np.array(max_feature[:32]))] + max_feature[32:]
    for i in range(32):
        data[:, :, i] /= max_feature[0]
    for i in range(32, 40):
        data[:, :, i] /= max_feature[i - 31]
    data = data[:, :8000, :]
    sz_data = np.shape(data)
    data = data.reshape(sz_data[0] * 200, 40, 40, 1)
    return data

def to_y(data):
    data.resize(len(data) * 40, 3)
    value_class = map(lambda x: 0 if x <= 3.0 else 1 if x <= 6.0 else 2,
            data[:, 4])
    unique, counts = np.unique(value_class, return_counts=True)
    print(dict(zip(unique, counts)))
    n_data = len(value_class)
    res = np.zeros((n_data, 3))
    res[np.arange(n_data), value_class] = 1
    # features were divided.
    return np.repeat(res, 200, axis = 0)


if __name__ == "__main__":
    pre_batch = '/data/klng/git/EvolutionaryDNN/Datasets/EEG_data/'
    whole_data = np.array([unpickle(pre_batch + 's%.2d.dat'%i) for i in range(1, 32 + 1)])
    # (256000, 40, 40)
    whole_x = to_x(np.array(map(lambda x: x["data"], whole_data)))
    # (256000, 3)
    whole_y = to_y(np.array(map(lambda x: x["labels"], whole_data)))
    print np.shape(whole_x)
    print np.shape(whole_y)
