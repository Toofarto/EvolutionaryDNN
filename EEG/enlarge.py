import numpy as np
import cPickle
import data_lib
import scipy.ndimage


def unpickle(file):
    fo = open(file, 'rb') 
    dict = cPickle.load(fo)
    fo.close()
    return dict

def to_x(data):
    dx = [0, 1, 1, 1, 0, -1, -1, -1]
    dy = [-1, -1, 0, 1, 1, 1, 0, -1]
    freq_bin = [(1, 3), (4, 7), (8, 13), (14, 30), (31, 50)]
    left_right = {1:17, 2:18, 3:20, 4:21, 5:22,
            6:23, 7:25, 8:26, 9:27, 10:28, 
            11:29, 12:30, 13:31, 14:32}
    location = {1: (2, 0), 2: (2, 1), 3:(2, 2), 4:(0, 2), 5:(1, 3),
            6:(3, 3), 7:(2, 4), 8:(0, 4), 9:(1, 5), 10: (3, 5),
            11:(2, 6), 12:(0, 6), 13:(2, 7), 14:(2, 8), 15:(4, 8), 
            16:(4, 6), 17:(6, 0), 18:(6, 1), 19:(4, 2), 20:(6, 2), 21:(8, 2), 22:(7, 3), 23:(5, 3), 24:(4, 4), 25:(6, 4), 
            26:(8, 4), 27:(7, 5), 28:(5, 5), 29:(6, 6), 30:(8, 6),
            31:(6, 7), 32:(6, 8)}

    data.resize(len(data) * 40, 40, 8064)
    data = data[:, :, 3*128:] # 7680
    sz_data = np.shape(data)
    res_data = np.zeros((sz_data[0] * 60, 9, 9, 5))
    for sample_idx in range(sz_data[0]):
        for channel_idx in range(32):
            n_span = 60 
            dst_x, dis_y = location[channel_idx + 1]
            for time_frame in range(n_span):
                FFT = np.fft.fft(data[sample_idx,
                    channel_idx, 
                    time_frame*128:(time_frame+1)*128])
                for freq_channel in range(5):
                    start, end = freq_bin[freq_channel]
                    freq = FFT[start: end+1]
                    DE = np.log(np.real(np.vdot(freq, freq)))
                    res_data[60*sample_idx + time_frame, dst_x, dis_y, freq_channel] = DE
    sz_resdata = np.shape(res_data)
    return res_data

def normalize(res_data):
    sz_resdata = np.shape(res_data)
    print sz_resdata
    maxs = np.max(np.abs(res_data), axis = 0)
    print "MAXS:"
    print maxs
    for y in range(sz_resdata[1]):
        for x in range(sz_resdata[2]):
            for ch_idx in range(sz_resdata[3]):
                if maxs[x, y, ch_idx] == 0: continue
                res_data[:, x, y, ch_idx] /= maxs[x, y, ch_idx]
    return res_data

def enlarge(data):
    sz_data = np.shape(data)
    factor = 3
    res_data = np.zeros((sz_data[0], sz_data[1]*factor, sz_data[2]*factor, 5), dtype=np.float32)
    for i in range(sz_data[0]):
        for ch in range(5):
            img = data[i, :, :, ch]
            res_img = scipy.ndimage.zoom(img, factor, order=1)
            res_data[i, :, :, ch] = res_img
    print res_data[10, :, :, :]
    return res_data

pre_batch = '/data/klng/git/EvolutionaryDNN/Datasets/EEG_data/'
whole_data = np.array([unpickle(pre_batch + 's%.2d.dat'%i) for i in range(1, 32 + 1)])
enlarged = enlarge(to_x(np.array(map(lambda x: x["data"], whole_data))))
print enlarged[1, :, :, :]
res_x = normalize(enlarge)
with open(pre_batch + 'graph_de_enlarge_normalize.dat', 'wb') as f:
    cPickle.dump(res_x, f)
