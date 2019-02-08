from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

import tensorflow as tf

from utils.metrics import AverageMeter
from utils.logger import DefinedSummarizer


class AttnTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger, data_loader):
        """
        Constructing the Attn trainer based on the Base Train..
        Here is the pipeline of constructing
        - Assign sess, model, config, logger, data_loader(if_specified)
        - Initialize all variables
        - Load the latest checkpoint
        - Create the summarizer
        - Get the nodes we will need to run it from the graph
        :param sess:
        :param model:
        :param config:
        :param logger:
        :param data_loader:
        """
        super(AttnTrainer, self).__init__(sess, model, config, logger, data_loader)

        # load the model from the latest checkpoint
        self.model.load(self.sess)

        # Summarizer
        self.summarizer = logger

        self.x_base, self.x_hour, self.y, self.is_training = tf.get_collection('inputs')
        self.train_op, self.loss_node, self.acc_node, self.acc2_node, self.acc3_node\
            , self.acc4_node, self.acc5_node, self.acc6_node, self.acc7_node, self.acc8_node\
            , self.acc9_node, self.acc10_node= tf.get_collection('train')
    
    def train(self):
        """
        This is the main loop of training
        Looping on the epochs
        :return:
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.test(cur_epoch)

    def train_epoch(self, epoch=None):
        """
        Train one epoch
        :param epoch: cur epoch number
        :return:
        """
        # initialize dataset
        self.data_loader.initialize(self.sess, is_train = True)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="epoch-{}-".format(epoch))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, acc= self.train_step()
            # update metrics returned from train_step func
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)

        self.sess.run(self.model.global_epoch_inc)

        # summarize
        summaries_dict = {'train/loss_per_epoch': loss_per_epoch.val,
                          'train/acc_per_epoch': acc_per_epoch.val}
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        self.model.save(self.sess)
        
        print("""
Epoch-{0}  loss:{1:.8f} -- acc:{2:.4f} -- lr:{3:.8f} -- exp_name:{4}
        """.format(epoch, loss_per_epoch.val, acc_per_epoch.val, self.model.learning_rate.val, self.config.exp_name))

        tt.close()

    def train_step(self):
        """
        Run the session of train_step in tensorflow
        also get the loss & acc of that minibatch.
        :return: (loss, acc) tuple of some metrics to be used in summaries
        """
        _, loss, acc= self.sess.run([self.train_op, self.loss_node, self.acc_node],
                                     feed_dict={self.is_training: True})
        return loss, acc
    
    def test(self, epoch):
        # initialize dataset
        self.data_loader.initialize(self.sess, is_train=False)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_test), total=self.data_loader.num_iterations_test,
                  desc="Val-{}-".format(epoch))

        loss_per_epoch = AverageMeter()
        acc_per_epoch = AverageMeter()
        acc2_per_epoch = AverageMeter()
        acc3_per_epoch = AverageMeter()
        acc4_per_epoch = AverageMeter()
        acc5_per_epoch = AverageMeter()
        acc6_per_epoch = AverageMeter()
        acc7_per_epoch = AverageMeter()
        acc8_per_epoch = AverageMeter()
        acc9_per_epoch = AverageMeter()
        acc10_per_epoch = AverageMeter()

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, acc, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10\
                = self.sess.run([self.loss_node, self.acc_node,
                                 self.acc2_node,self.acc3_node,self.acc4_node,self.acc5_node,
                                 self.acc6_node,self.acc7_node,self.acc8_node,self.acc9_node,self.acc10_node],
                                     feed_dict={self.is_training: False})
            # update metrics returned from train_step func
            loss_per_epoch.update(loss)
            acc_per_epoch.update(acc)
            acc2_per_epoch.update(acc2)
            acc3_per_epoch.update(acc3)
            acc4_per_epoch.update(acc4)
            acc5_per_epoch.update(acc5)
            acc6_per_epoch.update(acc6)
            acc7_per_epoch.update(acc7)
            acc8_per_epoch.update(acc8)
            acc9_per_epoch.update(acc9)
            acc10_per_epoch.update(acc10)

        # summarize
        summaries_dict = {'test/loss_per_epoch': loss_per_epoch.val,
                          'test/acc_per_epoch': acc_per_epoch.val}
        self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)
        
        print("""
Val-{}  loss:{:.8f} -- acc:{:.4f} -- acc2:{:.4f} -- acc3:{:.4f} -- acc4:{:.4f} -- acc5:{:.4f} -- acc6:{:.4f} -- acc7:{:.4f} -- acc8:{:.4f} -- acc9:{:.4f} -- acc10:{:.4f}
        """.format(epoch, loss_per_epoch.val, acc_per_epoch.val, acc2_per_epoch.val, acc3_per_epoch.val, acc4_per_epoch.val, acc5_per_epoch.val
                   , acc6_per_epoch.val, acc7_per_epoch.val, acc8_per_epoch.val, acc9_per_epoch.val, acc10_per_epoch.val))

        tt.close()