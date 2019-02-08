from base.base_model import BaseModel
import tensorflow as tf
import numpy as np


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
            self.x_base, self.x_hour, self.x_nsw, self.y = self.data_loader.get_input()
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.x_base)
        tf.add_to_collection('inputs', self.x_hour)
        tf.add_to_collection('inputs', self.x_nsw)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.is_training)

        """
        Network Architecture
        """
        with tf.variable_scope('network'):
            self.x_hour = tf.to_float(self.x_hour)
            with tf.variable_scope('inputall'):
                x_base_reshape = tf.reshape(self.x_base, [tf.shape(self.x_base)[0], tf.shape(self.x_base)[1], tf.shape(self.x_base)[2], 1]) # reshape base --> [?,6,172,1]
                conv = tf.contrib.layers.conv2d(
                    x_base_reshape,
                    1,
                    [self.x_base.shape[1],1],
                    stride=[self.x_base.shape[1],1],
                    padding='SAME',
                    data_format=None,
                    rate=1,
                    activation_fn=tf.nn.tanh,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope='base_conv'
                )

                conv = tf.squeeze(conv) # [?,1,172,1] --> [?,172]
                concat_input = tf.concat([self.x_hour, tf.reshape(self.x_nsw, [tf.shape(self.x_nsw)[0], 1]), conv], 1,
                                       name='concat_input')  # concat: [?,24],[?,],[?,172] --> [?,172+1+24]
                concat_input = tf.reshape(concat_input, [-1,172+1+24])

                fc1 = tf.contrib.layers.fully_connected(concat_input,
                                                        256,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc1")

                fc1 = tf.contrib.layers.batch_norm(fc1)

                fc2 = tf.contrib.layers.fully_connected(fc1,
                                                        256,
                                                        activation_fn=tf.nn.tanh,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="fc2")

                fc2 = tf.contrib.layers.batch_norm(fc2)

                self.out = tf.contrib.layers.fully_connected(fc2,
                                                        int(self.x_base.shape[2]),
                                                        activation_fn=tf.nn.softmax,
                                                        normalizer_fn=None,
                                                        normalizer_params=None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                        weights_regularizer=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                        biases_regularizer=None,
                                                        trainable=True,
                                                        scope="out")

                tf.add_to_collection('out', self.out)


        """
        Some operators for the training process
        """
        with tf.variable_scope('out_argmax'):
            self.out_argmax = tf.argmax(self.out, axis=1, output_type=tf.int64, name='out_argmax')  # --> [?,]

        with tf.variable_scope('loss-acc'):
            y_ind = tf.argmax(self.y, axis=1)  # --> [?,]
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.out, scope="loss")
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
            # no decay learning rate
            # starter_learning_rate = self.config.learning_rate
            # self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step_tensor,
            #                                            100000, 0.96, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
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
