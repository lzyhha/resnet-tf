from model import *
import tensorflow as tf


def configure():
    flags = tf.app.flags
    flags.DEFINE_integer('max_step', 2000, 'How many steps to train')
    flags.DEFINE_float('rate', 0.01, 'learning rate for training')
    flags.DEFINE_float('weight_decay', 1e-4, 'L2 regularization')
    flags.DEFINE_integer('reload_step', 4000, 'Reload step to continue training')
    flags.DEFINE_integer('save_interval', 100, 'interval to save model')
    flags.DEFINE_integer('summary_interval', 5, 'step to save summary')
    flags.DEFINE_integer('n_classes', 10, 'output class number')
    flags.DEFINE_integer('batch_size', 128, 'samples for one iter')
    flags.DEFINE_boolean('is_training', True, 'training or predict')
    flags.DEFINE_integer('layers', 2, 'res-net layers in a res-group')
    flags.DEFINE_string('logdir', 'logs', 'Log dir')
    flags.DEFINE_string('modeldir', 'models', 'Model dir')
    flags.DEFINE_string('model_name', 'ResNet', 'Model file name')

    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


if __name__ == '__main__':
    model = ResNet(configure(), tf.Session())
    model.train()
