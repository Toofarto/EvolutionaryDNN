from tf import *
from data_lib import *
from sklearn import metrics
import sys, cPickle, os


sz_x = 160
sz_y = 3
pre_batch = sys.argv[1]
n_GPU = int(sys.argv[2])
method_label = sys.argv[3]
chosen_class = map(int, sys.argv[4].split(','))
print "method_label:", method_label
print "chosen_class:", chosen_class

freq_bin = [(1, 3), (4, 7), (8, 13), (14, 30), (31, 50)]

def to_x(data):
    data.resize(len(data) * 40, 40, 8064)
    data = data[:, :, 3*128:]
    sz_data = np.shape(data)
    res_data = np.zeros((sz_data[0] * 60, 32 * 5))
    for sample_idx in range(sz_data[0]):
        for channel_idx in range(32):
            n_span = 60 
            for time_frame in range(n_span):
                FFT = np.fft.fft(data[sample_idx,
                    channel_idx, 
                    time_frame*128:(time_frame+1)*128])
                for freq_channel in range(5):
                    start, end = freq_bin[freq_channel]
                    freq = FFT[start: end+1]
                    res_data[sample_idx * n_span + time_frame, 
                            channel_idx * 5 + freq_channel] = np.log(np.real(np.vdot(freq, freq)))
    maxs = np.max(np.absolute(res_data), axis = 0)
    print("maxs length:", len(maxs))
    #mins = np.min(np.absolute(res_data), axis = 1)
    for i in range(len(maxs)):
        res_data[:, i] /= maxs[i]
    print("shape of to_x", np.shape(res_data))
    return res_data

def standardize(data):
    data = np.array(data)
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    res = np.array([(data[:, i] - mean[i])/std[i] for i in range(4)])
    return res

def normalize_y(data):
    data = np.array(data)
    maxs = np.max(data, axis = 0)
    mins = np.min(data, axis = 0)
    res = np.array([(data[:, i] - mins[i])/(maxs[i] - mins[i]) for i in range(4)])
    return res

y_count = {}
y_value = np.array([])

def to_y(data, method = ""):
    if method == "standardize": data = map(standardize, data)
    elif method == "normalize": data = map(normalize_y, data)
    data = np.array(data)
    sz_data = np.shape(data)
    n_repeat = 60
    data.resize(sz_data[0] * 40, 4)
    if method == "standardize": class_cutoff = [-0.43, 0.43]
    elif method == "normalize": class_cutoff = [0.333, 0.667]
    else: class_cutoff = [3, 6]
    value_class = map(lambda x: 
            0 if x <= class_cutoff[0] else 
            1 if x <= class_cutoff[1] else
            2, 
            data[:, 0])
    # return np.repeat(value_class, 63)
    unique, counts = np.unique(value_class, return_counts=True)
    global y_value
    y_value = np.repeat(np.array(value_class), n_repeat)
    y_count = dict(zip(unique, counts))
    print(dict(zip(unique, counts)))
    n_data = len(value_class)
    y_invfreq = [float(sum(list(y_count.values()))) / y_count[i] for i in range(3)]
    res = np.zeros((n_data, 3))
    res[np.arange(n_data), value_class] = 1
    # for i in range(3):
    #     res[:, i] *= y_invfreq[i]
    # features were divided.
    return np.repeat(res, n_repeat, axis = 0)

# Preprocess the data
time_start_preprocess = time.time()
whole_data = np.array([unpickle(pre_batch + 's%.2d.dat'%i) for i in range(1, 32 + 1)])
whole_y = to_y(map(lambda x: x["labels"], whole_data), method_label)

pre_x_filename = pre_batch +'freq.dat'
if os.path.isfile(pre_x_filename):
    whole_x = cPickle.load(open(pre_x_filename, 'rb'))
else:
    whole_x = to_x(np.array(map(lambda x: x["data"], whole_data)))
    cPickle.dump(whole_x, open(pre_x_filename, 'wb'))

cut_off = 28 * 40 * 60
#cut_off = int(len(whole_x) * float(28) / 32)
train_x = whole_x[:cut_off, :]
train_y = whole_y[:cut_off, :]
train_NP = NP_Dataset(train_x, train_y)

train_easy_x = [whole_x[(y_value == i)[:cut_off], :] for i in range(3)]
train_easy_y = [whole_y[(y_value == i)[:cut_off], :] for i in range(3)]
train_easy_x[0] = np.repeat(train_easy_x[0], 2, axis = 0)
train_easy_y[0] = np.repeat(train_easy_y[0], 2, axis = 0)
train_balaced_x = np.concatenate([train_easy_x[chosen_class[0]],
    train_easy_x[chosen_class[1]]])
train_balaced_y = np.concatenate([train_easy_y[chosen_class[0]], 
    train_easy_y[chosen_class[1]]])
train_balance_NP = NP_Dataset(train_balaced_x, train_balaced_y)

test_easy_x = [whole_x[(y_value == i)[cut_off:], :] for i in range(3)]
test_easy_y = [whole_y[(y_value == i)[cut_off:], :] for i in range(3)]
test_chosen_x = np.concatenate([test_easy_x[chosen_class[0]],
    test_easy_x[chosen_class[1]]])
test_chosen_y = np.concatenate([test_easy_y[chosen_class[0]], 
    test_easy_y[chosen_class[1]]])
test_chosen_NP = NP_Dataset(test_chosen_x, test_chosen_y)

train_easy_NP = [NP_Dataset(
    whole_x[(y_value == i)[:cut_off], :],
    whole_y[(y_value == i)[:cut_off], :]) 
    for i in range(3)]

test_NP = NP_Dataset(whole_x[cut_off:, :], whole_y[cut_off:, :])

print("train and test data precess finished")
print("time for preprocess", time.time() - time_start_preprocess)

def build_network():
    global x, y_, keep_prob
    x = tf.placeholder(tf.float32, shape=[None, sz_x], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[None, sz_y], name="y-input")
    keep_prob = tf.placeholder(tf.float32)

    # network = full_connect_layer(network, 2048, tf.nn.relu)
    # network = tf.nn.dropout(network, keep_prob)
    # network = full_connect_layer(x, 1024, tf.nn.relu)
    # network = tf.nn.dropout(network, keep_prob)
    # network = full_connect_layer(network, 512, tf.nn.relu)
    # network = tf.nn.dropout(network, keep_prob)
    fc1 = full_connect_layer(x , 256, tf.nn.relu)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)
    network = full_connect_layer(fc1_drop, sz_y, tf.nn.softmax)

    global cross_entropy, train_step, y_p, correct_prediction, accuracy
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(network), reduction_indices=[1]))
    train_step = tf.train.MomentumOptimizer(5e-6, 0.8).minimize(cross_entropy)
    y_p = tf.argmax(network, 1)
    correct_prediction = tf.equal(y_p, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if n_GPU > 0:
    for i in range(n_GPU):
        with tf.device("/gpu:%d"%i):
            build_network()
else :
    build_network()


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

start_train_time = time.time()
last_epoch = -1
data_NP = train_easy_NP[0]
#div = [700, 500, 500]
div = [0, 0, 0]
for i in range(300000):
    if i == sum(div[:1]):
        last_epoch = -1
        data_NP = train_easy_NP[1]
    if i == sum(div[:2]):
        last_epoch = -1
        data_NP = train_easy_NP[2]
    if i == sum(div[:3]):
        last_epoch = -1
        data_NP = train_balance_NP if method_label == "" else train_NP
    batch = data_NP.next_batch(32)
    if i%500 == 499:
        train_accuracy = accuracy.eval(feed_dict = {
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, epoch: %d, training accuracy %g"%(i+1, data_NP.get_epoch(), train_accuracy))
    elif data_NP.get_epoch() > last_epoch:
        test_NP = test_chosen_NP
        last_epoch = data_NP.get_epoch()
        test_batch = test_NP.next_batch(-1)
        print("finished epoch: %d"%last_epoch)
        gred, val_accuracy, y_pred = sess.run([cross_entropy, accuracy, y_p], feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0})
        print("test accuracy %g, gred: %g"%(val_accuracy, gred))
        y_true = np.argmax(test_batch[1], 1)
	print("Precision", metrics.precision_score(y_true, y_pred, average = "weighted"))
	print("Recall", metrics.recall_score(y_true, y_pred, average = "weighted"))
	print("f1_score", metrics.f1_score(y_true, y_pred, average = "weighted"))
	confusion_matrix = np.array(metrics.confusion_matrix(y_true, y_pred))
        p_arr = [float(confusion_matrix[i, i])/max(np.sum(confusion_matrix[:, i]), 1)
                for i in range(3)]
        r_arr = [float(confusion_matrix[i, i])/max(np.sum(confusion_matrix[i, :]), 1)
                for i in range(3)]
        f_arr = [((2 * p_arr[i] * r_arr[i])/(p_arr[i] + r_arr[i]) 
                if p_arr[i] + r_arr[i] != 0.0 else 0)
                for i in range(3)]
	print("confusion_matrix")
	print(confusion_matrix)
	print("precision_each_class", p_arr)
	print("recall_each_class", r_arr)
	print("f1_score", f_arr)
    else:  # Record a summary
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("Total Training Time:",(time.time() - start_train_time))
test_batch = test_NP.next_batch(-1)
final_test_accuracy = accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}) / test_time
print("final_test_accuracy = %f"%final_test_accuracy)


