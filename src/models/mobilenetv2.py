# Architecture based on MobileNetV2 https://arxiv.org/pdf/1801.04381.pdf
#https://github.com/ohadlights/mobilenetv2/blob/master/mobilenetv2.py
import tensorflow as tf
import tensorflow.contrib.slim as slim




def block(net, input_filters, output_filters, expansion, stride, scope=None, reuse=None):
    with tf.variable_scope(scope, 'block', [net], reuse=reuse):
        res_block = net
        res_block = slim.conv2d(inputs=res_block, num_outputs=input_filters * expansion, kernel_size=[1, 1], scope='conv2d_1')
        res_block = slim.separable_conv2d(inputs=res_block, num_outputs=None, kernel_size=[3, 3], stride=stride, scope='sepconv2d')
        res_block = slim.conv2d(inputs=res_block, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None, scope='conv2d_2')
        if stride == 2:
            net2 = res_block 
        else:
            if input_filters != output_filters:
                net = slim.conv2d(inputs=net, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None, scope='conv2d_3')
            net2 = tf.add(res_block, net)
    return net2


def blocks(net, expansion, output_filters, repeat, stride):
    input_filters = net.shape[3].value

    with tf.variable_scope('block_first'):
        # first layer should take stride into account
        net = block(net, input_filters, output_filters, expansion, stride)

    net = slim.repeat(net, repeat-1, block, input_filters, output_filters, expansion, 1)
    # for _ in range(1, repeat):
        # net = block(net, input_filters, output_filters, expansion, 1)

    return net

def inference_classify(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 depth_multiplier=1.0,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 scope='MobilenetV2'):

    endpoints = dict()

    expansion = 6

    with tf.variable_scope(scope):

        with slim.arg_scope(mobilenet_v2_arg_scope(0.0004, is_training=is_training, depth_multiplier=depth_multiplier,
                                                   dropout_keep_prob=dropout_keep_prob)):
            net = tf.identity(inputs)

            net = slim.conv2d(net, 32, [3, 3], scope='conv11', stride=2)

            net = blocks(net=net, expansion=1, output_filters=16, repeat=1, stride=1)

            net = blocks(net=net, expansion=expansion, output_filters=24, repeat=2, stride=2)

            net = blocks(net=net, expansion=expansion, output_filters=32, repeat=3, stride=2)

            net = blocks(net=net, expansion=expansion, output_filters=64, repeat=4, stride=2)

            net = blocks(net=net, expansion=expansion, output_filters=96, repeat=3, stride=1)

            net = blocks(net=net, expansion=expansion, output_filters=160, repeat=3, stride=2)

            net = blocks(net=net, expansion=expansion, output_filters=320, repeat=1, stride=1)

            net = slim.conv2d(net, 1280, [1, 1], scope='last_bottleneck')

            net = slim.avg_pool2d(net, [7, 7])

            logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='features')

            if spatial_squeeze:
                logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

            endpoints['Logits'] = logits

            if prediction_fn:
                endpoints['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, endpoints

# mobilenet_v2.default_image_size = 224


def mobilenet_v2_arg_scope(weight_decay, is_training=True, depth_multiplier=1.0, regularize_depthwise=False,
                           dropout_keep_prob=1.0):

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }

    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params
                        # normalizer_params={'is_training': is_training, 'center': True, 'scale': True }
                        ):

        with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):

            with slim.arg_scope([slim.separable_conv2d],
                                weights_regularizer=depthwise_regularizer, depth_multiplier=depth_multiplier):

                with slim.arg_scope([slim.dropout], is_training=is_training, keep_prob=dropout_keep_prob) as sc:

                    return sc

def inference(inputs,
                 dropout_keep_prob=0.999,
                 phase_train=True,
                 bottleneck_layer_size=512, 
                 weight_decay=0.0004,
                 reuse=None,
                 depth_multiplier=1.0,
                 spatial_squeeze=True):

    expansion = 6

    with tf.variable_scope('MobilenetV2', [inputs], reuse=reuse):

        with slim.arg_scope(mobilenet_v2_arg_scope(weight_decay, is_training=phase_train, depth_multiplier=depth_multiplier,
                                                   dropout_keep_prob=dropout_keep_prob)):

            net = slim.conv2d(inputs, 32, [3, 3], scope='conv11', stride=2)

            with tf.variable_scope('blocks1'):
                net = blocks(net=net, expansion=1, output_filters=16, repeat=1, stride=1)

            with tf.variable_scope('blocks2'):
                net = blocks(net=net, expansion=expansion, output_filters=24, repeat=2, stride=2)
            with tf.variable_scope('blocks3'):
                net = blocks(net=net, expansion=expansion, output_filters=32, repeat=3, stride=2)
            with tf.variable_scope('blocks4'):
                net = blocks(net=net, expansion=expansion, output_filters=64, repeat=4, stride=2)
            with tf.variable_scope('blocks5'):
                net = blocks(net=net, expansion=expansion, output_filters=96, repeat=3, stride=1)
            with tf.variable_scope('blocks6'):
                net = blocks(net=net, expansion=expansion, output_filters=160, repeat=3, stride=2)
            with tf.variable_scope('blocks7'):
                net = blocks(net=net, expansion=expansion, output_filters=320, repeat=1, stride=1)

            net = slim.conv2d(net, 1280, [1, 1], scope='last_bottleneck')

            net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avg_pool2d')

            logits = slim.conv2d(net, bottleneck_layer_size, [1, 1], activation_fn=None, normalizer_fn=None, reuse=False, scope='features') # differ

            if spatial_squeeze:
                logits = tf.squeeze(logits, [1, 2])

    return logits, None

