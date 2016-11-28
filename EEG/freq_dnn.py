from tf import *
from data_lib import *
from sklearn import metrics
import sys


sz_x = 160
sz_y = 3
pre_batch = sys.argv[1]
n_GPU = int(sys.argv[2])

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
    print np.shape(res_data)
    return res_data

y_count = {}

def to_y(data):
    data.resize(len(data) * 40, 4)
    value_class = map(lambda x: 0 if x <= 3.0 else 1 if x <= 6.0 else 2,
            data[:, 0])
    # return np.repeat(value_class, 63)
    unique, counts = np.unique(value_class, return_counts=True)
    y_count = dict(zip(unique, counts))
    print(dict(zip(unique, counts)))
    n_data = len(value_class)
    y_invfreq = [float(sum(list(y_count.values()))) / y_count[i] for i in range(3)]
    res = np.zeros((n_data, 3))
    res[np.arange(n_data), value_class] = 1
    for i in range(3):
        res[:, i] *= y_invfreq[i]
    # features were divided.
    return np.repeat(res, 60, axis = 0)

# Preprocess the data
time_start_preprocess = time.time()
whole_data = np.array([unpickle(pre_batch + 's%.2d.dat'%i) for i in range(1, 32 + 1)])
whole_y = to_y(np.array(map(lambda x: x["labels"], whole_data)))
whole_x = to_x(np.array(map(lambda x: x["data"], whole_data)))

cut_off = 28 * 40 * 60
train_x = whole_x[:cut_off, :]
train_y = whole_y[:cut_off, :]
train_NP = NP_Dataset(train_x, train_y)

test_x = whole_x[cut_off:, :]
test_y = whole_y[cut_off:, :]
test_NP = NP_Dataset(test_x, test_y)

print("train and test data precess finished")
print("time for preprocess", time.time() - time_start_preprocess)

def build_network():
    global x, y_, keep_prob
    x = tf.placeholder(tf.float32, shape=[None, sz_x], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[None, sz_y], name="y-input")
    keep_prob = tf.placeholder(tf.float32)

    network = full_connect_layer(x, 1024, tf.nn.relu)
    network = tf.nn.dropout(network, keep_prob)
    network = full_connect_layer(x, 512, tf.nn.relu)
    network = tf.nn.dropout(network, keep_prob)
    network = full_connect_layer(x, 256, tf.nn.relu)
    network = tf.nn.dropout(network, keep_prob)
    network = full_connect_layer(x, 128, tf.nn.relu)
    network = tf.nn.dropout(network, keep_prob)
    network = full_connect_layer(network, sz_y, tf.nn.softmax)

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
for i in range(300000):
    batch = train_NP.next_batch(64)
    if i%500 == 499:
        train_accuracy = accuracy.eval(feed_dict = {
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, epoch: %d, training accuracy %g"%(i+1, train_NP.get_epoch(), train_accuracy))
    elif train_NP.get_epoch() > last_epoch:
        last_epoch = train_NP.get_epoch()
        test_batch = test_NP.next_batch(8192)
        print("finished epoch: %d"%last_epoch)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0})
        print("test accuracy %g"%val_accuracy)
        y_true = np.argmax(test_batch[1], 1)
	# print("Precision", metrics.precision_score(y_true, y_pred))
	# print("Recall", metrics.recall_score(y_true, y_pred))
	# print("f1_score", metrics.f1_score(y_true, y_pred))
	print("confusion_matrix")
	print(metrics.confusion_matrix(y_true, y_pred))
    else:  # Record a summary
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("Total Training Time:",(time.time() - start_train_time))
test_time = 5
final_test_accuracy = 0.0
for i in range(test_time):
    test_batch = test_NP.next_batch(30000)
    final_test_accuracy += accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}) / test_time
print("final_test_accuracy = %f"%final_test_accuracy)
