from base.base_model import BaseModel
import tensorflow as tf
import numpy as np

ENTROPY_WEIGHT = 0.
ENTROPY_EPS = 1e-6

class Attn_Fuse_Model(BaseModel):
    def __init__(self, data_loader, config):
        super(Attn_Fuse_Model, self).__init__(config)
        # Get the data_loader to make the joint of the inputs in the graph
        self.data_loader = data_loader

        # define some important variables
        self.x_base = None  # input from 6 base recommender: (?,7,172)
        self.x_hour = None  # input from one-hot hour vector: (?,24)
        self.y = None  # output of label: one-hot channel vector: (?,172)
        self.is_training = None
        self.out_argmax = None
        self.loss = None
        self.acc = None
        self.optimizer = None
        self.train_step = None

        self.build_model()
        self.init_saver()

    def build_model(self):
        """
        Helper Variables
        """
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

        """
        Inputs to the network
        """
        with tf.variable_scope('inputs'):
            self.x_base, self.x_hour, self.x_nsw, self.x_seq, self.y = self.data_loader.get_input()
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.x_base)
        tf.add_to_collection('inputs', self.x_hour)
        tf.add_to_collection('inputs', self.x_nsw)
        tf.add_to_collection('inputs', self.x_seq)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.is_training)

        """
        Network Architecture
        """

        def RNN(x, weights, biases):
            # reshape to [1, n_input]
            n_input = self.config.batch_size
            print(x.shape)
            x = tf.reshape(x, [n_input, 172])
            # Generate a n_input-element sequence of inputs
            # (eg. [had] [a] [general] -> [20] [6] [33])
            x = tf.split(x,n_input,0)
            # 1-layer LSTM with n_hidden units.
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(256)
            # generate prediction
            outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
            # there are n_input outputs but
            # we only want the last output
            return tf.matmul(outputs[-1], weights['out']) + biases['out']

        conv_weights = {'out': tf.Variable(tf.random_normal([256, 6]))}
        conv_biases = {'out': tf.Variable(tf.random_normal([6]))}
        channel_weights = {'out': tf.Variable(tf.random_normal([256, 172]))}
        channel_biases = {'out': tf.Variable(tf.random_normal([172]))}

        with tf.variable_scope('network'):
            self.x_hour = tf.to_float(self.x_hour)
            self.x_nsw = tf.reshape(self.x_nsw,[tf.shape(self.x_nsw)[0],1])
            self.x_seq = tf.to_float(self.x_seq)   
            with tf.variable_scope('attn_base'):
                fc1_a = tf.contrib.layers.fully_connected(self.x_hour,
                                                        128,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc1_a")
                fc1_b = tf.contrib.layers.fully_connected(self.x_nsw,
                                                        128,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc1_b")

                # fc1_a = tf.contrib.layers.batch_norm(fc1_a)
                # fc1_b = tf.contrib.layers.batch_norm(fc1_b)

                fc2_a = tf.contrib.layers.fully_connected(fc1_a,
                                                        128,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc2_a")
                fc2_b = tf.contrib.layers.fully_connected(fc1_b,
                                                        128,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc2_b")


                # fc2_a = tf.contrib.layers.batch_norm(fc2_a)
                # fc2_b = tf.contrib.layers.batch_norm(fc2_b)

                fc3_a = tf.contrib.layers.fully_connected(fc2_a,
                                                        128,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc3_a")
                fc3_b = tf.contrib.layers.fully_connected(fc2_b,
                                                        128,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc3_b")

                # fc3_a = tf.contrib.layers.batch_norm(fc3_a,scope='bn_fc3_a')
                # fc3_b = tf.contrib.layers.batch_norm(fc3_b,scope='bn_fc3_b')

                conv_kernel_a = tf.contrib.layers.fully_connected(fc3_a,
                                                                int(self.x_base.shape[1]),
                                                                activation_fn=tf.nn.tanh,
                                                                normalizer_fn=None,
                                                                normalizer_params=None,
                                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                weights_regularizer=None,
                                                                biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                                biases_regularizer=None,
                                                                trainable=True,
                                                                scope="conv_kernel_a")
                conv_kernel_b = tf.contrib.layers.fully_connected(fc3_b,
                                                                int(self.x_base.shape[1]),
                                                                activation_fn=tf.nn.tanh,
                                                                normalizer_fn=None,
                                                                normalizer_params=None,
                                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                weights_regularizer=None,
                                                                biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                                biases_regularizer=None,
                                                                trainable=True,
                                                                scope="conv_kernel_b")

                conv_kernel_c = RNN(self.x_seq, conv_weights, conv_biases)
                conv_kernel_c = tf.identity(conv_kernel_c, name="conv_kernel_c")
                conv_kernel = tf.add_n([conv_kernel_a,conv_kernel_b,conv_kernel_c])

            with tf.variable_scope('attn_channel'):
                fc4_a = tf.contrib.layers.fully_connected(self.x_hour,
                                                        128,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc1_a")
                fc4_b = tf.contrib.layers.fully_connected(self.x_nsw,
                                                        128,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc1_b")

                # fc1_a = tf.contrib.layers.batch_norm(fc1_a)
                # fc1_b = tf.contrib.layers.batch_norm(fc1_b)

                fc5_a = tf.contrib.layers.fully_connected(fc4_a,
                                                        128,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc2_a")
                fc5_b = tf.contrib.layers.fully_connected(fc4_b,
                                                        128,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc2_b")


                # fc2_a = tf.contrib.layers.batch_norm(fc2_a)
                # fc2_b = tf.contrib.layers.batch_norm(fc2_b)

                fc6_a = tf.contrib.layers.fully_connected(fc5_a,
                                                        128,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc3_a")
                fc6_b = tf.contrib.layers.fully_connected(fc5_b,
                                                        128,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc3_b")

                # fc3_a = tf.contrib.layers.batch_norm(fc3_a,scope='bn_fc3_a')
                # fc3_b = tf.contrib.layers.batch_norm(fc3_b,scope='bn_fc3_b')

                channel_mask_a = tf.contrib.layers.fully_connected(fc6_a,
                                                                172,
                                                                activation_fn=tf.nn.tanh,
                                                                normalizer_fn=None,
                                                                normalizer_params=None,
                                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                weights_regularizer=None,
                                                                biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                                biases_regularizer=None,
                                                                trainable=True,
                                                                scope="conv_kernel_a")
                channel_mask_b = tf.contrib.layers.fully_connected(fc6_b,
                                                                172,
                                                                activation_fn=tf.nn.tanh,
                                                                normalizer_fn=None,
                                                                normalizer_params=None,
                                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                weights_regularizer=None,
                                                                biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                                biases_regularizer=None,
                                                                trainable=True,
                                                                scope="conv_kernel_b")
                
                channel_mask_c = RNN(self.x_seq, channel_weights, channel_biases)
                channel_mask_c = tf.identity(channel_mask_c, name="channel_mask_c")
                channel_mask = tf.add_n([channel_mask_a,channel_mask_b,channel_mask_c])

            with tf.variable_scope('attn_fuse'):
                conv_kernel = tf.reshape(conv_kernel, [tf.shape(conv_kernel)[0], 1, -1], name="conv_kernel_reshape")
                prod = tf.matmul(conv_kernel, tf.stop_gradient(self.x_base), name="prod_matmul")  # [?,1,6]*[?,6,172] --> [?,1,172]
                self.out = tf.squeeze(prod, name="out_before_mask")  # [?,172]
                self.out = tf.add(self.out, channel_mask, name="out_after_mask")  # [?,172]
                # self.out = tf.nn.softmax(self.out, name="out_softmax")

                # self.out = tf.nn.softmax(prod, name="out") #-->[?,172]

                # x_base_reshape = tf.reshape(self.x_base, [tf.shape(self.x_base)[0], tf.shape(self.x_base)[1], tf.shape(self.x_base)[2], 1]) # reshape base --> [?,6,172,1]
                # conv_kernel_reshape = tf.reshape(conv_kernel, [tf.shape(conv_kernel)[0], tf.shape(conv_kernel)[1], 1, 1]) # reshape kernel --> [?,6,1,1]
                # conv = tf.nn.conv2d(x_base_reshape, conv_kernel_reshape, [1, 6, 1, 1], "SAME", name = "Conv")
                # self.out = tf.squeeze(conv) #-->[?,172]
                # self.out = tf.contrib.layers.softmax(self.out)

                tf.add_to_collection('out', self.out)

        """
        Some operators for the training process
        """
        with tf.variable_scope('out_argmax'):
            self.out_argmax = tf.argmax(self.out, axis=1, output_type=tf.int64, name='out_argmax')  # --> [?,]

        with tf.variable_scope('loss-acc'):
            y_ind = tf.argmax(self.y, axis=1)  # --> [?,]
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.stop_gradient(self.y), logits=self.out, scope="loss")
            self.entropy = ENTROPY_WEIGHT * -1. * tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.nn.sigmoid(self.out),tf.log(tf.nn.sigmoid(self.out) + ENTROPY_EPS)),axis=1))
            self.loss_entropy = self.loss + self.entropy
            # self.loss_log = tf.log(tf.nn.sigmoid(self.out) + ENTROPY_EPS)
            # self.loss_mul = tf.multiply(tf.nn.sigmoid(self.out),tf.log(tf.nn.sigmoid(self.out) + ENTROPY_EPS))
            # self.red_sum = tf.reduce_mean(tf.multiply(tf.nn.sigmoid(self.out),tf.log(tf.nn.sigmoid(self.out) + ENTROPY_EPS)),axis=1)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(y_ind, self.out_argmax), tf.float32), name="acc")
            self.acc2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 2), tf.float32), name="acc2")
            self.acc3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 3), tf.float32), name="acc3")
            self.acc4 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 4), tf.float32), name="acc4")
            self.acc5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 5), tf.float32), name="acc5")
            self.acc6 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 6), tf.float32), name="acc6")
            self.acc7 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 7), tf.float32), name="acc7")
            self.acc8 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 8), tf.float32), name="acc8")
            self.acc9 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 9), tf.float32), name="acc9")
            self.acc10 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 10), tf.float32), name="acc10")
            self.acc20 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 20), tf.float32), name="acc20")
            self.acc30 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 30), tf.float32), name="acc30")
            self.acc40 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 40), tf.float32), name="acc40")
            self.acc50 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.out, y_ind, 50), tf.float32), name="acc50")


        with tf.variable_scope('train_step'):
            # starter_learning_rate = self.config.learning_rate
            # self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step_tensor,
            #                                            100000, 0.96, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss_entropy, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.entropy)
        tf.add_to_collection('train', self.loss_entropy)
        # tf.add_to_collection('train', self.loss_log)
        # tf.add_to_collection('train', self.loss_mul)
        # tf.add_to_collection('train', self.red_sum)
        tf.add_to_collection('train', self.acc)
        tf.add_to_collection('train', self.acc2)
        tf.add_to_collection('train', self.acc3)
        tf.add_to_collection('train', self.acc4)
        tf.add_to_collection('train', self.acc5)
        tf.add_to_collection('train', self.acc6)
        tf.add_to_collection('train', self.acc7)
        tf.add_to_collection('train', self.acc8)
        tf.add_to_collection('train', self.acc9)
        tf.add_to_collection('train', self.acc10)
        tf.add_to_collection('train', self.acc20)
        tf.add_to_collection('train', self.acc30)
        tf.add_to_collection('train', self.acc40)
        tf.add_to_collection('train', self.acc50)

        print("total num of weights: ", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    def init_saver(self):
        """
        initialize the tensorflow saver that will be used in saving the checkpoints.
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)
