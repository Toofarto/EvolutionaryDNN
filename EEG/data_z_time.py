from data_lib import *


egg_pos = {1: (0, 3), 2: (1, 2), 3: (2, 2), 4: (2, 0), 5: (3, 1), 6: (3, 3), 7: (4, 2), 8: (4, 0), 9: (5, 1), 10: (5, 3), 11: (6, 2), 12: (6, 0), 13: (7, 2), 14: (8, 3), 15: (8, 4), 16: (9, 4), 17: (0, 5), 18: (1, 6), 19: (2, 4), 20: (2, 6), 21: (2, 8), 22: (3, 7), 23: (3, 5), 24: (4, 4), 25: (4, 6), 26: (4, 8), 27: (5, 7), 28: (5, 5), 29: (6, 6), 30: (6, 8), 31: (7, 6), 32: (8, 5)}

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

    thickness = 10
    res_data = np.zeros((sz_data[0] * (8000/thickness), 10, 9, thickness))
    for i in range(sz_data[0]):
        for j in range(8000):
            for k in range(32):
                dst_y, dst_x = egg_pos[k+1]
                res_data[i * (8000/thickness) + (j/thickness), dst_y, dst_x, j%thickness] = data[i, j, k];

    return res_data

def to_y(data):
    data.resize(len(data) * 40, 4)
    value_class = map(lambda x: 0 if x <= 3.0 else 1 if x <= 6.0 else 2,
            data[:, 0])
    unique, counts = np.unique(value_class, return_counts=True)
    print(dict(zip(unique, counts)))
    n_data = len(value_class)
    res = np.zeros((n_data, 3))
    res[np.arange(n_data), value_class] = 1
    # features were divided.
    return np.repeat(res, 800, axis = 0)


if __name__ == "__main__":
    pre_batch = '/data/klng/git/EvolutionaryDNN/Datasets/EEG_data/'
    whole_data = np.array([unpickle(pre_batch + 's%.2d.dat'%i) for i in range(1, 32 + 1)])
    # (256000, 40, 40)
    whole_x = to_x(np.array(map(lambda x: x["data"], whole_data)))
    # (256000, 3)
    whole_y = to_y(np.array(map(lambda x: x["labels"], whole_data)))
    print np.shape(whole_x)
    print np.shape(whole_y)
