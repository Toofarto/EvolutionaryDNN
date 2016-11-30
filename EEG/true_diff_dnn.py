from tf import *
from data_lib import *
from sklearn import metrics
import sys, cPickle, os

sz_x = 160
sz_y = 3
n_hidden_1 = 256
pre_batch = sys.argv[1]
n_GPU = int(sys.argv[2])
worker_name = sys.argv[3]


def to_y(data):
    data = np.array(data)
    sz_data = np.shape(data)
    n_repeat = 60
    data.resize(sz_data[0] * 40, 4)
    class_cutoff = [3, 6]
    value_class = map(lambda x: 
            0 if x <= class_cutoff[0] else 
            1 if x <= class_cutoff[1] else
            2, 
            data[:, 0])
    unique, counts = np.unique(value_class, return_counts=True)
    global y_value
    y_value = np.repeat(np.array(value_class), n_repeat)
    y_count = dict(zip(unique, counts))
    print(y_count)
    n_data = len(value_class)
    res = np.zeros((n_data, 3))
    res[np.arange(n_data), value_class] = 1
    return np.repeat(res, n_repeat, axis = 0)

def to_x(data):
    freq_bin = [(1, 3), (4, 7), (8, 13), (14, 30), (31, 50)]

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
    for i in range(len(maxs)):
        res_data[:, i] /= maxs[i]
    return res_data

# Preprocess the data
time_start_preprocess = time.time()
whole_data = np.array([unpickle(pre_batch + 's%.2d.dat'%i) for i in range(1, 32 + 1)])
whole_y = to_y(map(lambda x: x["labels"], whole_data))

pre_x_filename = pre_batch +'freq.dat'
if os.path.isfile(pre_x_filename):
    whole_x = cPickle.load(open(pre_x_filename, 'rb'))
else:
    whole_x = to_x(np.array(map(lambda x: x["data"], whole_data)))
    cPickle.dump(whole_x, open(pre_x_filename, 'wb'))

cut_off = [25 * 40 * 60, 29 * 40 * 60]
#cut_off = int(len(whole_x) * float(28) / 32)
train_x = whole_x[:cut_off[0], :]
train_y = whole_y[:cut_off[0], :]
train_NP = NP_Dataset(train_x, train_y)

train_easy_x = [whole_x[(y_value == i) & (np.arange(len(whole_x)) < cut_off[0]), :] for i in range(3)]
train_easy_y = [whole_y[(y_value == i) & (np.arange(len(whole_y)) < cut_off[0]), :] for i in range(3)]
train_easy_x[0] = np.repeat(train_easy_x[0], 2, axis = 0)
train_easy_y[0] = np.repeat(train_easy_y[0], 2, axis = 0)
train_balaced_x = np.concatenate(train_easy_x)
train_balaced_y = np.concatenate(train_easy_y)
train_balance_NP = NP_Dataset(train_balaced_x, train_balaced_y)

test_NP = NP_Dataset(whole_x[cut_off[0]:cut_off[1], :], whole_y[cut_off[0]:cut_off[1]:, :])
validation_NP = NP_Dataset(whole_x[cut_off[1]:, :], whole_y[cut_off[1]:, :])

print("train and test data preprocess finished")
print("time for preprocess", time.time() - time_start_preprocess)

class Model_Evaluation(object):
    def __init__(self, val_accuracy, y_pred, y_true, gred):
        self.accuracy = val_accuracy
        self.y_predict = y_pred
        self.y_true = y_true
        self.gradient = gred

        self.avg_precision =  metrics.precision_score(self.y_true, self.y_predict, average = "weighted")
        self.avg_recall = metrics.recall_score(self.y_true, self.y_predict, average = "weighted")
        self.avg_f1 = metrics.f1_score(self.y_true, self.y_predict, average = "weighted")
        self.confusion_matrix = np.array(metrics.confusion_matrix(self.y_true, self.y_predict))
        self.p_arr = [float(self.confusion_matrix[i, i])/max(np.sum(self.confusion_matrix[:, i]), 1)
                for i in range(3)]
        self.r_arr = [float(self.confusion_matrix[i, i])/max(np.sum(self.confusion_matrix[i, :]), 1)
                for i in range(3)]
        self.f_arr = [((2 * self.p_arr[i] * self.r_arr[i])/(self.p_arr[i] + self.r_arr[i]) if (self.p_arr[i] + self.r_arr[i]) != 0.0 else 0)
                for i in range(3)]

    def display(self):
        print "=============Start=================="
        print "Confusion Matrix:"
        print self.confusion_matrix
        print "Average Accuracy:", self.accuracy
        print "Average Precision:", self.avg_precision
        print "Average Recall:", self.avg_recall
        print "Average F1:", self.avg_f1
        print "=============End===================="

class SimpleDNNModel(object):
    def __init__(self, sz_x, sz_y, n_GPU, name = ""):
        self.sz_x = sz_x
        self.sz_y = sz_y
        self.n_GPU = n_GPU
        self.graph = tf.Graph()
        self.name = name

        self.beta = 0.1
        self.FC1 = FullConnectLayer()
        # self.load()

    def load(self, index_model = 0):
        with self.graph.as_default():
            if self.n_GPU > 0:
                with tf.device("/gpu:%d"%(index_model%n_GPU)):
                    self.build_network()
            else :
                self.build_network()

    def build_network(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.sz_x], name="x-input")
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.sz_y], name="y-input")
        self.keep_prob = tf.placeholder(tf.float32)

        self.FC1.load(self.x, n_hidden_1, tf.nn.relu)
        fc1_drop = tf.nn.dropout(self.FC1.layer, self.keep_prob)
        network = full_connect_layer(fc1_drop, self.sz_y, tf.nn.softmax)
        # network = full_connect_layer(self.FC1.layer, self.sz_y, tf.nn.softmax)

        self.cross_entropy = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(tf.log(network), self.y_) + 
            self.beta*tf.nn.l2_loss(self.FC1.Weight) + 
            self.beta*tf.nn.l2_loss(self.FC1.Bias) )
        # learning rate
        self.train_step = tf.train.AdadeltaOptimizer(5e-5).minimize(self.cross_entropy)
        self.y_p = tf.argmax(network, 1)
        self.correct_prediction = tf.equal(self.y_p, tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def fitting(self, train_NP, test_NP, train_batch = 32, test_batch = -1, max_step = 1000000):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.initialize_all_variables())

            start_train_time = time.time()
            last_epoch = -1
            for i in range(max_step):
                batch = train_NP.next_batch(32)
                if i%500 == 499:
                    self.train_accuracy = self.accuracy.eval(feed_dict = {self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0}, session=sess)
                    #print("step %d, epoch: %d, training accuracy %g"%(i+1, train_NP.get_epoch(), train_accuracy))
                elif train_NP.get_epoch() > last_epoch:
                    last_epoch = train_NP.get_epoch()
                    test_batch = test_NP.next_batch(-1)
                    gred, val_accuracy, y_pred = sess.run([self.cross_entropy, self.accuracy, self.y_p], 
                            feed_dict={self.x: test_batch[0], self.y_: test_batch[1], self.keep_prob: 1.0})
                else:  # Record a summary
                    self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

            # print("Total Training Time:",(time.time() - start_train_time))
            test_batch = test_NP.next_batch(-1) 
            gred, val_accuracy, y_pred = sess.run([self.cross_entropy, self.accuracy, self.y_p], 
                    feed_dict={self.x: test_batch[0], self.y_: test_batch[1], self.keep_prob: 1.0})
            y_true = np.argmax(test_batch[1], 1)
            self.test_evaluation = Model_Evaluation(val_accuracy, y_pred, y_true, gred)

            validation_batch = validation_NP.next_batch(128)
            v_gred, v_accuracy, v_y_pred = sess.run([self.cross_entropy, self.accuracy, self.y_p], 
                    feed_dict={self.x: validation_batch[0], self.y_: validation_batch[1], self.keep_prob: 1.0})
            self.fitness_evaluation = Model_Evaluation(v_accuracy, v_y_pred, y_true, v_gred)
            self.fitness = self.fitness_evaluation.avg_f1

            self.FC1.update(*sess.run([self.FC1.Weight, self.FC1.Bias]))

n_model = 20
n_generation = 100
crossover_rate = 0.4
change_prop = 0.2
max_step = 70000
Models = [SimpleDNNModel(sz_x, sz_y, n_GPU) for i in range(n_model)]
time_start_generation = time.time()
for i in range(len(Models)):
    model = Models[i]
    model.load()
    model.fitting(train_balance_NP, test_NP, max_step = max_step)
    model.test_evaluation.display()
Models.sort(key = lambda x:x.fitness, reverse=True)
Models[0].test_evaluation.display()
print "Start Different Evolve"
for iter_gen in range(n_generation):
    #Different Evolve
    for i in range(len(Models)):
        a, b, c = np.random.choice(np.delete(np.arange(len(Models)), i), 3)
        if random.uniform(0, 1) < crossover_rate:
            weight_a, bias_a = Models[a].FC1.get()
            weight_b, bias_b = Models[b].FC1.get()
            weight_c, bias_c = Models[c].FC1.get()
            weight_y = weight_a + change_prop * (weight_b - weight_c)
            bias_y = bias_a + change_prop * (bias_b - bias_c)
            model_y = SimpleDNNModel(sz_x, sz_y, n_GPU)
            model_y.FC1.update(weight_y, bias_y)
            model_y.load()
            model_y.fitting(train_balance_NP, test_NP, max_step = max_step)
            if model_y.fitness > Models[i].fitness: Models[i] = model_y
    print "Generation: %d, train accuracy: %f, test accuracy: %f, fitness: %f"%(iter_gen, model.train_accuracy, model.test_evaluation.accuracy, model.fitness)
    Models.sort(key = lambda x:x.fitness, reverse=True)
    print "n_model: %d, n_generation: %d, crossover_rate: %d, max_step: %d, n_hidden_1: %d"%(n_model, n_generation, crossover_rate, max_step, n_hidden_1)
    print "#%d, Best fitness (accuracy of validation): %f"%(iter_gen, Models[0].fitness)
    Models[0].test_evaluation.display()
    print "Time spent on this generation", time.time() - time_start_generation
    time_start_generation = time.time()
