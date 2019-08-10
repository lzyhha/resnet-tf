from utils import *
import tensorflow as tf
import time
from datetime import timedelta
import numpy as np
from input import *
import math


class ResNet(object):

    def __init__(self, conf, sess):
        self.conf = conf
        self.sess = sess

        self.images = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='image')
        self.labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label')
        self.y = self.residual_net()

        self.loss_op, self.softmax = self.loss()
        self.acc, self.predict = self.accuracy()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = self.train_op()

        self.sess.run(tf.global_variables_initializer())

        trainable_vars = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        self.saver = tf.train.Saver(var_list=trainable_vars + bn_moving_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, graph=self.sess.graph)

        self.summary = self.config_summary()

    def config_summary(self):
        summarys = [tf.summary.scalar('loss', self.loss_op),
                    tf.summary.scalar('accuracy', self.acc)]
        summary = tf.summary.merge(summarys)
        return summary

    def save_summary(self, summary, step):
        print('summarizing', end=' ')
        self.writer.add_summary(summary, step)

    def save(self, step):
        print('saving', end=' ')
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('------- no such checkpoint', model_path)
            return False
        self.saver.restore(self.sess, model_path)
        return True

    def residual_net(self, scope='ResNet'):
        with tf.variable_scope(scope):
            y = conv(self.images, kernel_size=3, output_channel=64, stride_size=1)
            y = batch_norm(y, training=self.conf.is_training, scope='bn_begin')
            y = tf.nn.relu(y)

            y = residual_group(y, layers=self.conf.layers, output_channel=64,
                               subsample=False, is_training=self.conf.is_training,
                               scope='group1')
            y = residual_group(y, layers=self.conf.layers, output_channel=128,
                               subsample=True, is_training=self.conf.is_training,
                               scope='group2')
            y = residual_group(y, layers=self.conf.layers, output_channel=256,
                               subsample=True, is_training=self.conf.is_training,
                               scope='group3')
            y = residual_group(y, layers=self.conf.layers, output_channel=512,
                               subsample=True, is_training=self.conf.is_training,
                               scope='group4')
            y = conv(y, kernel_size=3, output_channel=1024, stride_size=1)
            y = batch_norm(y, training=self.conf.is_training, scope='bn_end')
            y = tf.nn.relu(y)
            y = tf.nn.avg_pool2d(y, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1],
                                 padding='VALID')
            y = tf.squeeze(y, axis=[1, 2])
            y = tf.layers.dense(y, units=self.conf.n_classes)
        return y

    def loss(self, scope='loss'):
        with tf.variable_scope(scope):
            targets = tf.one_hot(self.labels, depth=self.conf.n_classes, axis=-1, name='one-hot')
            softmax = tf.nn.softmax(self.y)
            entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.y,
                                                                   labels=targets,
                                                                   name='cross-entropy')
            entropy_loss = tf.reduce_mean(entropy_loss, name='loss')

            tf.add_to_collection('losses', entropy_loss)

            weight_l2_losses = [tf.nn.l2_loss(o) for o in tf.get_collection('weights')]
            weight_decay_loss = tf.multiply(self.conf.weight_decay, tf.add_n(weight_l2_losses),
                                            name='weight_decay_loss')
            tf.add_to_collection('losses', weight_decay_loss)

            loss_op = tf.add_n(tf.get_collection('losses'), name='loss_op')
        tf.summary.scalar(name='losses', tensor=loss_op)
        return loss_op, softmax

    def accuracy(self, scope='accuracy'):
        with tf.variable_scope(scope):
            preds = tf.argmax(self.y, -1)
            acc = 1.0 - tf.nn.zero_fraction(
                tf.cast(tf.equal(preds, self.labels), dtype=tf.int32))
        return acc, preds

    def train_op(self):
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss_op, params, name='gradients')
        optimizer = tf.train.MomentumOptimizer(self.conf.rate, 0.9)
        update = optimizer.apply_gradients(zip(gradients, params))
        with tf.control_dependencies([update]):
            train_op = tf.no_op(name='train_op')
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.rate).minimize(self.loss_op)
        return train_op

    def train(self):
        if self.conf.reload_step > 0:
            if not self.reload(self.conf.reload_step):
                return
            print('reload', self.conf.reload_step)

        images, labels = distorted_inputs('cifar', self.conf.batch_size)

        tf.train.start_queue_runners(sess=self.sess)

        print('Begin Train')
        for train_step in range(1, self.conf.max_step+1):
            start_time = time.time()

            x, y = self.sess.run([images, labels])

            # summary
            if train_step == 1 or train_step % self.conf.summary_interval == 0:
                feed_dict = {self.images: x,
                             self.labels: y}
                loss, acc, _, summary = self.sess.run(
                    [self.loss_op, self.acc, self.optimizer, self.summary],
                    feed_dict=feed_dict)
                print(str(train_step), '----Training loss:', loss, ' accuracy:', acc, end=' ')
                self.save_summary(summary, train_step+self.conf.reload_step)
            # print 损失和准确性
            else:
                feed_dict = {self.images: x,
                             self.labels: y}
                loss, acc, _ = self.sess.run(
                    [self.loss_op, self.acc, self.optimizer], feed_dict=feed_dict)
                print(str(train_step), '----Training loss:', loss, ' accuracy:', acc, end=' ')
            # 保存模型
            if train_step % self.conf.save_interval == 0:
                self.save(train_step+self.conf.reload_step)
            end_time = time.time()
            time_diff = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))

    def eval(self):
        if self.conf.reload_step > 0:
            if not self.reload(self.conf.reload_step):
                return
            print('reload', self.conf.reload_step)

        images, labels = inputs(True, 'cifar', self.conf.batch_size)

        tf.train.start_queue_runners(sess=self.sess)

        print('Begin Eval')
        true_count = 0
        num_iter = int(math.ceil(10000 / self.conf.batch_size))
        total_count = num_iter * self.conf.batch_size

        top_k_op = tf.nn.in_top_k(self.softmax, self.labels, 1)

        for train_step in range(1, num_iter + 1):
            x, y = self.sess.run([images, labels])
            feed_dict = {self.images: x,
                         self.labels: y}
            preds = self.sess.run([top_k_op], feed_dict=feed_dict)
            true_count += np.sum(preds)

        acc = true_count/total_count
        print(acc)

