import tensorflow as tf


def conv(inputs, kernel_size, output_channel,
         stride_size=1, init_bias=0.1,
         padding='SAME', stddev=0.01,
         scope='conv'):
    with tf.variable_scope(scope):
        input_channel = inputs.get_shape().as_list()[-1]
        weights = tf.Variable(tf.random.truncated_normal(
            shape=[kernel_size, kernel_size, input_channel, output_channel],
            stddev=stddev,
            dtype=tf.float32))
        tf.add_to_collection('weights', weights)
        biases = tf.Variable(tf.constant(init_bias, shape=[output_channel], dtype=tf.float32))
        conv_layer = tf.nn.conv2d(inputs, weights, [1, stride_size, stride_size, 1],
                                  padding=padding)
        conv_layer = tf.nn.bias_add(conv_layer, biases)
    return conv_layer


def batch_norm(inputs, training, scope='bn'):
    with tf.variable_scope(scope):
        bn = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            momentum=0.99,
            epsilon=1e-3,
            center=True,
            scale=True,
            training=training,
            fused=True)

    return bn


def residual_block(inputs, output_channel,
                   subsample, is_training=True, scope='res_block'):
    with tf.variable_scope(scope):
        if subsample:
            y = conv(inputs, kernel_size=3,
                     output_channel=output_channel, stride_size=2,
                     scope='conv1')
            shortcut = conv(inputs, kernel_size=1,
                            output_channel=output_channel,
                            stride_size=2, scope='shortcut')
        else:
            y = conv(inputs, kernel_size=3,
                     output_channel=output_channel,
                     stride_size=1, scope='conv1')
            shortcut = tf.identity(inputs, name='short1')
        y = batch_norm(y, training=is_training, scope='bn1')
        shortcut = batch_norm(shortcut, training=is_training, scope='bn_s')
        y = tf.nn.relu(y, name='relu1')
        y = conv(y, kernel_size=3, output_channel=output_channel, stride_size=1,
                 scope='conv2')
        y = batch_norm(y, training=is_training, scope='bn2')
        y = y + shortcut
        y = tf.nn.relu(y, name='relu2')
    return y


def residual_group(inputs, layers, output_channel, subsample,
                   is_training=True, scope='res_group'):
    with tf.variable_scope(scope):
        y = residual_block(inputs, output_channel=output_channel,
                           subsample=subsample, is_training=is_training,
                           scope='block1')
        for i in range(layers - 1):
            y = residual_block(y, output_channel=output_channel, subsample=False,
                               is_training=is_training,
                               scope='block%d' % (i+2))
    return y
