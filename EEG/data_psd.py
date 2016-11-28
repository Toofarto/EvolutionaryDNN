from data_lib import *


egg_pos = {1: (0, 3), 2: (1, 2), 3: (2, 2), 4: (2, 0), 5: (3, 1), 6: (3, 3), 7: (4, 2), 8: (4, 0), 9: (5, 1), 10: (5, 3), 11: (6, 2), 12: (6, 0), 13: (7, 2), 14: (8, 3), 15: (8, 4), 16: (9, 4), 17: (0, 5), 18: (1, 6), 19: (2, 4), 20: (2, 6), 21: (2, 8), 22: (3, 7), 23: (3, 5), 24: (4, 4), 25: (4, 6), 26: (4, 8), 27: (5, 7), 28: (5, 5), 29: (6, 6), 30: (6, 8), 31: (7, 6), 32: (8, 5)}

freq_bin = [(1, 3), (4, 7), (8, 13), (14, 30), (31, 50)]

def to_x(data):
    data.resize(len(data) * 40, 40, 8064)
    sz_data = np.shape(data)
    res_data = np.zeros((sz_data[0] * 63 * 5, 10, 9, 1))
    for sample_idx in range(sz_data[0]):
        for channel_idx in range(32):
            dst_y, dst_x = egg_pos[channel_idx+1]
            n_span = 8064/128
            for time_frame in range(n_span):
                FFT = np.fft.fft(data[sample_idx,
                    channel_idx, 
                    time_frame*128:(time_frame+1)*128])
                for freq_channel in range(5):
                    start, end = freq_bin[freq_channel]
                    freq = FFT[start: end+1]
                    res_data[sample_idx * n_span + time_frame * 5 + freq_channel, 
                            dst_y, 
                            dst_x, 0] = np.vdot(freq, freq)
    print np.shape(res_data)
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
    return np.repeat(res, 63, axis = 0)


if __name__ == "__main__":
    pre_batch = '/data/klng/git/EvolutionaryDNN/Datasets/EEG_data/'
    whole_data = np.array([unpickle(pre_batch + 's%.2d.dat'%i) for i in range(1, 32 + 1)])
    # (256000, 40, 40)
    whole_x = to_x(np.array(map(lambda x: x["data"], whole_data)))
    # (256000, 3)
    whole_y = to_y(np.array(map(lambda x: x["labels"], whole_data)))
    print np.shape(whole_x)
    print np.shape(whole_y)
