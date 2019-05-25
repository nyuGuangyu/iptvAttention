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
        self.channel_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 26, 27, 28, 29, 30,
                              31, 32, 33, 34,
                              35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65,
                              66, 67, 68, 69, 70,
                              73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                              96, 97, 98, 99,
                              100, 101, 102, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                              121, 122, 123,
                              124, 125, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
                              147, 148, 149,
                              150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
                              168, 169, 171,
                              201, 701, 702, 703, 704, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813,
                              814, 815, 816,
                              817, 818, 903, 905, 907, 908]
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

        with tf.variable_scope('network'):
            self.x_hour = tf.to_float(self.x_hour)
            self.x_nsw = tf.to_float(self.x_nsw)
            self.x_seq = tf.to_float(self.x_seq)
            with tf.variable_scope('attn_base'):
                fc1_a = tf.contrib.layers.fully_connected(self.x_hour,
                                                          32,
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
                                                          32,
                                                          activation_fn=tf.nn.tanh,
                                                          normalizer_fn=None,
                                                          normalizer_params=None,
                                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                          weights_regularizer=None,
                                                          biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                          biases_regularizer=None,
                                                          trainable=True,
                                                          scope="fc1_b")

                fc1_a = tf.contrib.layers.batch_norm(fc1_a)
                fc1_b = tf.contrib.layers.batch_norm(fc1_b)

                fc2_a = tf.contrib.layers.fully_connected(fc1_a,
                                                          32,
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
                                                          32,
                                                          activation_fn=tf.nn.tanh,
                                                          normalizer_fn=None,
                                                          normalizer_params=None,
                                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                          weights_regularizer=None,
                                                          biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                          biases_regularizer=None,
                                                          trainable=True,
                                                          scope="fc2_b")

                fc2_a = tf.contrib.layers.batch_norm(fc2_a)
                fc2_b = tf.contrib.layers.batch_norm(fc2_b)

                fc3_a = tf.contrib.layers.fully_connected(fc2_a,
                                                          32,
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
                                                          32,
                                                          activation_fn=tf.nn.tanh,
                                                          normalizer_fn=None,
                                                          normalizer_params=None,
                                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                          weights_regularizer=None,
                                                          biases_initializer=tf.contrib.layers.xavier_initializer(),
                                                          biases_regularizer=None,
                                                          trainable=True,
                                                          scope="fc3_b")

                fc3_a = tf.contrib.layers.batch_norm(fc3_a)
                fc3_b = tf.contrib.layers.batch_norm(fc3_b)

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

                conv_kernel = tf.add_n([conv_kernel_a, conv_kernel_b])

            with tf.variable_scope('attn_channel'):


            with tf.variable_scope('attn_fuse'):
                conv_kernel = tf.reshape(conv_kernel, [tf.shape(conv_kernel)[0], 1, -1], name="conv_kernel_reshape")
                prod = tf.matmul(conv_kernel, tf.stop_gradient(self.x_base),
                                 name="prod_matmul")  # [?,1,6]*[?,6,172] --> [?,1,172]
                self.out = tf.squeeze(prod, name="out_before_mask")  # [?,172]
                self.out = tf.add(self.out, channel_mask_c, name="out_after_mask")  # [?,172]

                tf.add_to_collection('out', self.out)

        """
        Some operators for the training process
        """
        with tf.variable_scope('out_argmax'):
            self.out_argmax = tf.argmax(self.out, axis=1, output_type=tf.int64, name='out_argmax')  # --> [?,]

        with tf.variable_scope('loss-acc'):
            y_ind = tf.argmax(self.y, axis=1)  # --> [?,]
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.stop_gradient(self.y), logits=self.out,
                                                        scope="loss")
            self.entropy = ENTROPY_WEIGHT * -1. * tf.reduce_mean(
                tf.reduce_sum(tf.multiply(tf.nn.sigmoid(self.out), tf.log(tf.nn.sigmoid(self.out) + ENTROPY_EPS)),
                              axis=1))
            self.loss_entropy = self.loss + self.entropy
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
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            # self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
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
