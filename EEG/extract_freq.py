import numpy as np

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

freq_bin = [(1, 3), (4, 7), (8, 13), (14, 30), (31, 50)]

def to_x(data):
    data.resize(len(data) * 40, 40, 8064)
    sz_data = np.shape(data)
    res_data = np.zeros((sz_data[0] * 63, 32 * 5))
    for sample_idx in range(sz_data[0]):
        for channel_idx in range(32):
            n_span = 8064/128
            for time_frame in range(n_span):
                FFT = np.fft.fft(data[sample_idx,
                    channel_idx, 
                    time_frame*128:(time_frame+1)*128])
                for freq_channel in range(5):
                    start, end = freq_bin[freq_channel]
                    freq = FFT[start: end+1]
                    res_data[sample_idx * n_span + time_frame , 
                            channel_idx * 5 + freq_channel] = np.log(np.real(np.vdot(freq, freq)))
    print np.shape(res_data)
    return res_data

def to_y(data):
    data.resize(len(data) * 40, 4)
    value_class = map(lambda x: 0 if x <= 3.0 else 1 if x <= 6.0 else 2,
            data[:, 0])
    return np.repeat(value_class, 63)
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
    whole_x = to_x(np.array(map(lambda x: x["data"], whole_data)))
    whole_y = to_y(np.array(map(lambda x: x["labels"], whole_data)))
    with open('/data/klng/git/EvolutionaryDNN/EEG/deap_freq.arff', 'w') as f:
        for (x, y) in zip(whole_x, whole_y):
            f.writelines("%d, %s\n"%(y, ",".join([str(i) for i in list(x)])))
