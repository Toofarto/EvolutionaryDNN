import time

from EEG_DATA_LIB import *


egg_pos = {1: (0, 3), 2: (1, 2), 3: (2, 2), 4: (2, 0), 5: (3, 1), 6: (3, 3), 7: (4, 2), 8: (4, 0), 9: (5, 1), 10: (5, 3), 11: (6, 2), 12: (6, 0), 13: (7, 2), 14: (8, 3), 15: (8, 4), 16: (9, 4), 17: (0, 5), 18: (1, 6), 19: (2, 4), 20: (2, 6), 21: (2, 8), 22: (3, 7), 23: (3, 5), 24: (4, 4), 25: (4, 6), 26: (4, 8), 27: (5, 7), 28: (5, 5), 29: (6, 6), 30: (6, 8), 31: (7, 6), 32: (8, 5)}

def to_x(data):
    sz_group = 10
    data.resize(len(data) * 40, 40, 8064)
    data = data.swapaxes(1, 2)
    data = data[:, :8000, :]
    # normalize
    max_feature = [max(abs(np.max(data[:, :, i])), abs(np.min(data[:, :, i]))) for i in range(40)]
    # the fist 32 channel is EE of the brain, regards as image
    max_feature = [np.max(np.array(max_feature[:32]))] + max_feature[32:]
    for i in range(32):
        data[:, :, i] /= max_feature[0]
    for i in range(32, 40):
        data[:, :, i] /= max_feature[i - 31]
    sz_data = np.shape(data)
    res_data = np.zeros((sz_data[0] * 320, 50, 45, 1))
    for i in range(sz_data[0]):
        for j in range(0, sz_data[1], 25):
            for k in range(5):
                for m in range(5):
                    for p in range(32):
                        dst_y, dst_x = egg_pos[p+1]
                        res_data[i * 320 + (j / 25), k * 10 + dst_y, m * 9 + dst_x, 0] = data[i, j + k * 5 + m, p]
    return res_data

def to_y(data):
    data.resize(len(data) * 40, 4)
    value_class = map(lambda x: 0 if x <= 3.0 else 1 if x <= 6.0 else 2,
            data[:, 0])
    n_data = len(value_class)
    res = np.zeros((n_data, 3))
    res[np.arange(n_data), value_class] = 1
    # features were divided.
    return np.repeat(res, 320, axis = 0)

if __name__ == "__main__":
    pre_batch = '/data/klng/git/EvolutionaryDNN/Datasets/EEG_data/'
    time_start_whole_x = time.time()
    whole_data = np.array([unpickle(pre_batch + 's%.2d.dat'%i) for i in range(1, 32 + 1)])
    whole_x = to_x(np.array(map(lambda x: x["data"], whole_data)))
    time_start_whole_y = time.time()
    print("Time for whole x: ", time_start_whole_y - time_start_whole_x)
    whole_y = to_y(np.array(map(lambda x: x["labels"], whole_data)))
    time_finish_process = time.time()
    print("Time for whole x: ", time_finish_process - time_start_whole_y)
    print np.shape(whole_x)
    print np.shape(whole_y)
