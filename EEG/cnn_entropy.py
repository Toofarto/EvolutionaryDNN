from extended_tf import *
from data_lib import *
from sklearn import metrics
import sys, cPickle, os

n_repeat = 60
pre_batch = sys.argv[1]
n_GPU = int(sys.argv[2])
worker_name = sys.argv[3]

def to_y(data):
    data = np.array(data)
    sz_data = np.shape(data)
    data.resize(sz_data[0] * 40, 4)
    class_cutoff = [3, 6]
    value_class = map(lambda x: 0 if x <= class_cutoff[0] else 
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

# Preprocess the data
time_start_preprocess = time.time()
whole_data = np.array([unpickle(pre_batch + 's%.2d.dat'%i) for i in range(1, 32 + 1)])
whole_y = to_y(map(lambda x: x["labels"], whole_data))

# features
pre_x_filename = pre_batch + 'graph_de_left_normalized.dat'
if os.path.isfile(pre_x_filename):
    whole_x = cPickle.load(open(pre_x_filename, 'rb'))
else:
    print "Can't find the features file"
    # whole_x = to_x(np.array(map(lambda x: x["data"], whole_data)))
    # cPickle.dump(whole_x, open(pre_x_filename, 'wb'))

cut_off = [25 * 40 * 60, 29 * 40 * 60]
# cut_off = [25 * 40 * 60, 29 * 40 * 60]
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

validation_NP = NP_Dataset(whole_x[cut_off[0]:cut_off[1], :], whole_y[cut_off[0]:cut_off[1]:, :])
test_NP = NP_Dataset(whole_x[cut_off[1]:, :], whole_y[cut_off[1]:, :])

print("train and test data preprocess finished")
print("time for preprocess", time.time() - time_start_preprocess)

class Model_Evaluation(object):
    def __init__(self, val_accuracy, y_pred, y_true, gred, softmax = -1):
        self.accuracy = val_accuracy
        self.y_predict = y_pred
        self.y_true = y_true
        self.gradient = gred
        self.softmax = softmax

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
        print "Loss:", self.gradient
        print "=============End===================="

def weight_variable(shape, sz = 20.0):
    initial = tf.random_normal(shape, stddev=math.sqrt(2.0/sz))
    return tf.Variable(initial, name="weight")

def bias_variable(shape):
    # initial = tf.constant(0.0, shape=shape)
    initial = tf.zeros(shape)
    return tf.Variable(initial, name="bias")

class SimpleDNNModel(object):
    def __init__(self, rate, name = ""):
        self.graph = tf.Graph()
        self.name = name
        self.rate = rate

        self.beta = 0.01

        self.load()

    def load(self, index_model = 0):
        with self.graph.as_default():
            if n_GPU > 0:
                with tf.device("/gpu:%d"%(index_model%n_GPU)):
                    self.build_network()
            else :
                self.build_network()

    def build_network(self):
        # self.x = tf.placeholder(tf.float32, shape=[None, 27, 27, 5], name="x-input")
        self.x = tf.placeholder(tf.float32, shape=[None, 9, 9, 5], name="x-input")
        self.y_ = tf.placeholder(tf.float32, shape=[None, 3], name="y-input")
        self.keep_prob = tf.placeholder(tf.float32)

	offset = self.x + tf.ones(shape=tf.pack(tf.shape(self.x)))

        network = offset

        self.cv1, w11 = conv_layer(self.x, 64, 3, lrelu)
        network = tf.nn.dropout(network, self.keep_prob)
        network, w12 = conv_layer(network, 64, 3, lrelu)
        network = self.cv1
        network = max_pool_2x2(network)

        network, w21  = conv_layer(network, 128, 3, lrelu)
        network = tf.nn.dropout(network, self.keep_prob)
        network, w22 = conv_layer(network, 128, 3, lrelu)
        network = max_pool_2x2(network)

        network, w31 = conv_layer(network, 256, 3, lrelu)
        network = tf.nn.dropout(network, self.keep_prob)
        network, w32 = conv_layer(network, 256, 3, lrelu)
        network = max_pool_2x2(network)

        network = combine(network, offset)
        network, w41, b41 = full_connect_layer(network, 2048, lrelu)
        network = tf.nn.dropout(network, self.keep_prob)
        network, w42, b42  = full_connect_layer(network, 2048, lrelu)
        network = tf.nn.dropout(network, self.keep_prob)
        network, w43, b43 = full_connect_layer(network, 3, tf.nn.softmax)

        # self.cross_entropy = tf.reduce_sum(-tf.reduce_mean(self.y_ * tf.log(network), reduction_indices=[1]))
        self.softmax = network
        self.cross_entropy = tf.reduce_sum(-tf.reduce_mean(self.y_ * tf.log(network), reduction_indices=[1]))
                #self.beta*(w11 + w21 + w31 + w41 + b41 + w42 + b42 + w43 + b43))
                #self.beta*(w11 + w12 + w21 + w22 + w31 + w32 +w41 + b41 + w42 + b42 + w43 + b43))
        # learning rate
        self.train_step = tf.train.AdamOptimizer(self.rate).minimize(self.cross_entropy)
        self.y_p = tf.argmax(network, 1)
        self.correct_prediction = tf.equal(self.y_p, tf.argmax(self.y_, 1)) 
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32)) 

    def fitting(self, fit_train_NP, fit_test_NP, train_batch = 8, test_batch = -1, max_step = 1000000):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.initialize_all_variables())

            start_train_time = time.time()
            last_epoch = -1
            for i in range(max_step):
                batch = fit_train_NP.next_batch(32)
                if i%500 == 499:
                    train_acc = sess.run([self.accuracy], 
                            feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                    print "train_acc:", train_acc
                    validation_batch = validation_NP.next_batch(32)
                    v_gred, v_accuracy, v_y_pred, v_softmax, v_cv1 = sess.run([self.cross_entropy, self.accuracy, self.y_p, self.softmax, self.cv1], 
                            feed_dict={self.x: validation_batch[0], self.y_: validation_batch[1], self.keep_prob: 1.0})
                    print "softmax of validation:", (v_softmax)
                    print "sv1:", np.count_nonzero(v_cv1), np.shape(v_cv1)
                    sum_vacc, n_iter = 0.0, 37
                    for i in range(n_iter):
                        validation_batch = validation_NP.next_batch(256)
                        v_y_true = np.argmax(validation_batch[1], 1)
                        v_gred, v_accuracy, v_y_pred = sess.run([self.cross_entropy, self.accuracy, self.y_p], 
                                feed_dict={self.x: validation_batch[0], self.y_: validation_batch[1], self.keep_prob: 1.0})
                        self.fitness_evaluation = Model_Evaluation(v_accuracy, v_y_pred, v_y_true, v_gred)
                        self.fitness = self.fitness_evaluation.avg_f1
                        sum_vacc += v_accuracy
                    print("step %d, epoch: %d, validation accuracy %g"%(i+1, fit_train_NP.get_epoch(), sum_vacc/n_iter))
                    # gred, val_accuracy, y_pred, test_softmax, cv1 = sess.run([self.cross_entropy, self.accuracy, self.y_p, self.softmax, self.cv1], 
                    #         feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                    # print("step %d, epoch: %d, training accuracy %g"%(i+1, fit_train_NP.get_epoch(), val_accuracy))
                    # print "cv1 nonzero:", np.shape(np.nonzero(cv1))
                elif fit_train_NP.get_epoch() > last_epoch:
                    last_epoch = fit_train_NP.get_epoch()
                    test_batch = fit_test_NP.next_batch(512)
                    gred, val_accuracy, y_pred, test_softmax = sess.run([self.cross_entropy, self.accuracy, self.y_p, self.softmax], 
                            feed_dict={self.x: test_batch[0], self.y_: test_batch[1], self.keep_prob: 1.0})
                    #print "Test Accuracy: ", val_accuracy
                    y_true = np.argmax(test_batch[1], 1)
                    self.test_evaluation = Model_Evaluation(val_accuracy, y_pred, y_true, gred, softmax = test_softmax)
                    self.test_evaluation.display()
                else:  # Record a summary
                    self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

	    sum_acc, n_iter = 0.0, 37
	    for i in range(n_iter):
		test_batch = fit_test_NP.next_batch(256) 
		y_true = np.argmax(test_batch[1], 1)
		gred, val_accuracy, y_pred, test_softmax = sess.run([self.cross_entropy, self.accuracy, self.y_p, self.softmax], 
			feed_dict={self.x: test_batch[0], self.y_: test_batch[1], self.keep_prob: 1.0})
		self.test_evaluation = Model_Evaluation(val_accuracy, y_pred, y_true, gred, softmax = test_softmax)
		sum_acc += val_accuracy
	    print sum_acc/n_iter


n_span = 1
max_step = 900000
#try_rate = np.linspace(1e-8, 1e2)
for i in range(n_span):
    #rate = try_e[i]
    model = SimpleDNNModel(5e-7)
    model.fitting(train_balance_NP, test_NP, max_step = max_step)
    # print "rate: ", rate
    print "test accuracy: %f, fitness: %f"%(model.test_evaluation.accuracy, model.fitness)
    model.test_evaluation.display()
