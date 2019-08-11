from model import *
import tensorflow as tf


def configure():
    flags = tf.app.flags
    flags.DEFINE_integer('max_step', 2000, 'How many steps to train')
    flags.DEFINE_float('rate', 0.01, 'learning rate for training')
    flags.DEFINE_float('weight_decay', 1e-4, 'L2 regularization')
    flags.DEFINE_integer('reload_step', 4000, 'Reload step to continue training')
    flags.DEFINE_integer('save_interval', 100, 'interval to save model')
    flags.DEFINE_integer('summary_interval', 5, 'interval to save summary')
    flags.DEFINE_integer('n_classes', 10, 'output class number')
    flags.DEFINE_integer('batch_size', 128, 'batch size for one iter')
    flags.DEFINE_boolean('is_training', True, 'training or predict (for batch normalization)')
    flags.DEFINE_integer('layers', 2, 'number of res-net layers in a res-group')
    flags.DEFINE_string('datadir', 'cifar', 'directory of data')
    flags.DEFINE_string('logdir', 'logs', 'directory to save logs of accuracy and loss')
    flags.DEFINE_string('modeldir', 'models', 'directory to save models ')
    flags.DEFINE_string('model_name', 'ResNet', 'Model file name')

    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


if __name__ == '__main__':
    model = ResNet(configure(), tf.Session())
    model.train()
