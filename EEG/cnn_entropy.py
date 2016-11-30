from tf import *
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
pre_x_filename = pre_batch + 'graph_de_normalized.dat'
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
train_easy_x[0] = np.repeat(train_easy_x[0], 1, axis = 0)
train_easy_y[0] = np.repeat(train_easy_y[0], 1, axis = 0)
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
    def __init__(self, rate, name = ""):
        self.graph = tf.Graph()
        self.name = name
        self.rate = rate

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

        network = conv_layer(self.x, 64, 3, tf.nn.relu)
        network = tf.nn.dropout(network, self.keep_prob)
        network = conv_layer(network, 64, 3, tf.nn.relu)
        network = max_pool_2x2(network)

        network = conv_layer(network, 128, 3, tf.nn.relu)
        network = tf.nn.dropout(network, self.keep_prob)
        network = conv_layer(network, 128, 3, tf.nn.relu)
        network = max_pool_2x2(network)

        network = conv_layer(network, 256, 3, tf.nn.relu)
        network = tf.nn.dropout(network, self.keep_prob)
        network = conv_layer(network, 256, 3, tf.nn.relu)
        network = max_pool_2x2(network)

        network = full_connect_layer(network, 1024, tf.nn.relu)
        network = tf.nn.dropout(network, self.keep_prob)
        network = full_connect_layer(network, 1024, tf.nn.relu)
        network = tf.nn.dropout(network, self.keep_prob)
        network = full_connect_layer(network, 3, tf.nn.softmax)

        #self.cross_entropy = tf.reduce_sum(-tf.reduce_mean(self.y_ * tf.log(network), reduction_indices=[1]))
        all_one = tf.Variable(tf.constant([1.0, 1.0, 1.0]))
        self.cross_entropy = tf.reduce_sum(tf.reduce_sum((all_one - self.y_) * tf.log(network), reduction_indices=[1]))
        # learning rate
        self.train_step = tf.train.AdamOptimizer(self.rate).minimize(self.cross_entropy)
        self.y_p = tf.argmax(network, 1)
        self.correct_prediction = tf.equal(self.y_p, tf.argmax(self.y_, 1)) 
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32)) 

    def fitting(self, train_NP, test_NP, train_batch = 32, test_batch = -1, max_step = 1000000):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.initialize_all_variables())

            start_train_time = time.time()
            last_epoch = -1
            for i in range(max_step):
                batch = train_NP.next_batch(train_batch)
                if i%500 == 499:
                    self.train_accuracy = self.accuracy.eval(feed_dict = {self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0}, session=sess)
                    print("step %d, epoch: %d, training accuracy %g"%(i+1, train_NP.get_epoch(), self.train_accuracy))
                elif train_NP.get_epoch() > last_epoch:
                    last_epoch = train_NP.get_epoch()
                    test_batch = test_NP.next_batch(128)
                    gred, val_accuracy, y_pred = sess.run([self.cross_entropy, self.accuracy, self.y_p], 
                            feed_dict={self.x: test_batch[0], self.y_: test_batch[1], self.keep_prob: 1.0})
                    #print "Test Accuracy: ", val_accuracy
                    y_true = np.argmax(test_batch[1], 1)
                    self.test_evaluation = Model_Evaluation(val_accuracy, y_pred, y_true, gred)
                    self.test_evaluation.display()
                else:  # Record a summary
                    self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

            test_batch = test_NP.next_batch(256) 
            y_true = np.argmax(test_batch[1], 1)
            gred, val_accuracy, y_pred = sess.run([self.cross_entropy, self.accuracy, self.y_p], 
                    feed_dict={self.x: test_batch[0], self.y_: test_batch[1], self.keep_prob: 1.0})
            self.test_evaluation = Model_Evaluation(val_accuracy, y_pred, y_true, gred)

            validation_batch = validation_NP.next_batch(256)
            v_y_true = np.argmax(validation_batch[1], 1)
            v_gred, v_accuracy, v_y_pred = sess.run([self.cross_entropy, self.accuracy, self.y_p], 
                    feed_dict={self.x: validation_batch[0], self.y_: validation_batch[1], self.keep_prob: 1.0})
            self.fitness_evaluation = Model_Evaluation(v_accuracy, v_y_pred, v_y_true, v_gred)
            self.fitness = self.fitness_evaluation.avg_f1


n_span = 1
max_step = 300000
#try_rate = np.linspace(1e-8, 1e2)
for i in range(n_span):
    #rate = try_e[i]
    model = SimpleDNNModel(5e-6)
    model.fitting(train_balance_NP, test_NP, max_step = max_step)
    # print "rate: ", rate
    print "train accuracy: %f, test accuracy: %f, fitness: %f"%(model.train_accuracy, model.test_evaluation.accuracy, model.fitness)
    model.test_evaluation.display()
