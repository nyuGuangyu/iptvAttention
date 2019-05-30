import numpy
import tensorflow as tf
import numpy as np
import cPickle as cp
from sklearn.preprocessing import MinMaxScaler
from collections import Counter



class DataGenerator:
    def __init__(self, config):
        self.config = config

        with open(self.config.data_train, "rb") as f:
            self.data_train = cp.load(f)
        with open(self.config.data_test, "rb") as f1:
            self.data_test = cp.load(f1)

        self.x_train_base = self.data_train['x_train_base'] # [?,7,172] last row is title sim
        self.x_train_hour = self.data_train['x_train_hour']
        self.x_train_nsw = self.data_train['x_train_nsw']
        self.x_train_item = self.data_train['x_train_item']
        self.x_train_seq = self.data_train['x_train_lstm']
        self.y_train = self.data_train['y_train']

        self.x_test_base = self.data_test['x_test_base']
        self.x_test_hour = self.data_test['x_test_hour']
        self.x_test_nsw = self.data_test['x_test_nsw']
        self.x_test_item = self.data_test['x_test_item']
        self.x_test_seq = self.data_test['x_test_lstm']
        self.y_test = self.data_test['y_test']

        # try normalize x_train_base, x_test_base, x_train_nsw, x_test_nsw
        v = self.x_train_base
        v_min = self.x_train_base.min(axis=2, keepdims=True)
        v_max = self.x_train_base.max(axis=2, keepdims=True)
        self.x_train_base = (v - v_min) / (v_max - v_min)
        where_are_NaNs = np.isnan(self.x_train_base)
        self.x_train_base[where_are_NaNs] = 0.

        v = self.x_test_base
        v_min = self.x_test_base.min(axis=2, keepdims=True)
        v_max = self.x_test_base.max(axis=2, keepdims=True)
        self.x_test_base = (v - v_min) / (v_max - v_min)
        where_are_NaNs = np.isnan(self.x_test_base)
        self.x_test_base[where_are_NaNs] = 0.

        print('x_train_base: ', self.x_train_base.shape, self.x_train_base.dtype)
        print('x_train_hour: ', self.x_train_hour.shape, self.x_train_hour.dtype)
        print('x_train_nsw: ', self.x_train_nsw.shape, self.x_train_nsw.dtype)
        print('x_train_item: ', self.x_train_item.shape, self.x_train_item.dtype)
        print('x_train_seq: ', self.x_train_seq.shape, self.x_train_seq.dtype)
        print('y_train: ', self.y_train.shape, self.y_train.dtype)
        assert self.x_train_base.shape[0] == self.x_train_hour.shape[0]
        assert self.x_train_hour.shape[0] == self.y_train.shape[0]
        assert self.y_train.shape[0] == self.x_train_nsw.shape[0]

        print('x_test_base: ', self.x_test_base.shape, self.x_test_base.dtype)
        print('x_test_hour: ', self.x_test_hour.shape, self.x_test_hour.dtype)
        print('x_test_nsw: ', self.x_test_nsw.shape, self.x_test_nsw.dtype)
        print('x_test_item: ', self.x_test_item.shape, self.x_test_item.dtype)
        print('x_test_seq: ', self.x_test_seq.shape, self.x_test_seq.dtype)
        print('y_test: ', self.y_test.shape, self.y_test.dtype)
        assert self.x_test_base.shape[0] == self.x_test_hour.shape[0]
        assert self.x_test_hour.shape[0] == self.y_test.shape[0]
        assert self.y_test.shape[0] == self.x_test_nsw.shape[0]

        self.train_len = self.x_train_base.shape[0]
        self.test_len = self.x_test_base.shape[0]

        self.num_iterations_train = (self.train_len + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_test = (self.test_len + self.config.batch_size - 1) // self.config.batch_size
        print("Data loaded successfully..")

        self.features_placeholder = None
        self.labels_placeholder = None

        self.dataset = None
        self.iterator = None
        self.init_iterator_op = None
        self.next_batch = None

        self.build_dataset_api()

    def build_dataset_api(self):
        with tf.device('/cpu:0'):
            self.base_placeholder = tf.placeholder(tf.float32, [None] + list(self.x_train_base.shape[1:]))
            self.hour_placeholder = tf.placeholder(tf.int64, [None] + list(self.x_train_hour.shape[1:]))
            self.nsw_placeholder = tf.placeholder(tf.int64, [None] + list(self.x_train_nsw.shape[1:])) # number of switch is a one hot vec
            self.item_placeholder = tf.placeholder(tf.int64, [None] + list(self.x_train_item.shape[1:]))
            self.seq_placeholder = tf.placeholder(tf.int64, [None] + list(self.x_train_seq.shape[1:]))
            self.labels_placeholder = tf.placeholder(tf.int64, [None] + list(self.y_train.shape[1:]))

            self.dataset = tf.data.Dataset.from_tensor_slices((self.base_placeholder, self.hour_placeholder, self.nsw_placeholder, self.item_placeholder, self.seq_placeholder, self.labels_placeholder))
            self.dataset = self.dataset.batch(self.config.batch_size)

            self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types,
                                                            self.dataset.output_shapes)

            self.init_iterator_op = self.iterator.make_initializer(self.dataset)

            self.next_batch = self.iterator.get_next()

            print("X_base_batch shape dtype: ", self.next_batch[0].shape)
            print("X_hour_batch shape dtype: ", self.next_batch[1].shape)
            print('X_nsw_batch shape dtype: ', self.next_batch[2].shape)
            print("Y_batch shape dtype: ", self.next_batch[3].shape)

    def initialize(self, sess, is_train):
        if is_train:
            idx = np.random.choice(self.train_len, self.train_len, replace=False)
            self.x_train_base = self.x_train_base[idx] # shuffle the data set
            self.x_train_hour = self.x_train_hour[idx]
            self.x_train_nsw = self.x_train_nsw[idx]
            self.x_train_item = self.x_train_item[idx]
            self.x_train_seq = self.x_train_seq[idx]
            self.y_train = self.y_train[idx]
            sess.run(self.init_iterator_op, feed_dict={self.base_placeholder: self.x_train_base,
			                                           self.hour_placeholder: self.x_train_hour,
                                                       self.nsw_placeholder: self.x_train_nsw,
                                                       self.item_placeholder: self.x_train_item,
                                                       self.seq_placeholder: self.x_train_seq,
                                                       self.labels_placeholder: self.y_train})
        else:
            sess.run(self.init_iterator_op, feed_dict={self.base_placeholder: self.x_test_base,
                                                       self.hour_placeholder: self.x_test_hour,
                                                       self.nsw_placeholder: self.x_test_nsw,
                                                       self.item_placeholder: self.x_test_item,
                                                       self.seq_placeholder: self.x_test_seq,
                                                       self.labels_placeholder: self.y_test})

    def get_input(self):
        # return self.next_batch
        return self.iterator.get_next()

    def get_holder(self):
        return self.base_placeholder, self.hour_placeholder, self.nsw_placeholder, self.seq_placeholder, self.labels_placeholder