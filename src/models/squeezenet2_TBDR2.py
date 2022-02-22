from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# https://github.com/liujuanLT/squeezenet

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d
from tensorflow.contrib.layers import batch_norm, l2_regularizer
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
import tensorflow.contrib.slim as slim

@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with arg_scope([conv2d, max_pool2d]):
            net = _squeeze(inputs, squeeze_depth)
            net = _expand(net, expand_depth)
        return net


def _squeeze(inputs, num_outputs):
    return conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def _expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat([e1x1, e3x3], 3)


class Squeezenet(object):
    """Original squeezenet architecture for 224x224 images."""
    name = 'squeezenet'

    def __init__(self, args):
        self._num_classes = args['num_classes']
        self._weight_decay = args['weight_decay']
        self._batch_norm_decay = args['batch_norm_decay']
        self._is_built = False
        self.bottleneck_layer_size = args['bottleneck_layer_size']
        self.dropout_keep_prob = args['dropout_keep_prob']

    def build(self, x, is_training):
        self._is_built = True
        with tf.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(is_training,
                                      self._weight_decay,
                                      self._batch_norm_decay)):
                return self._squeezenet(x, self._num_classes, self.bottleneck_layer_size, self.dropout_keep_prob)

    @staticmethod
    def _squeezenet(images, num_classes=1000, bottleneck_layer_size=512, dropout_keep_prob=0.8):
        net = conv2d(images, 96, [7, 7], stride=2, scope='conv1')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
        net = fire_module(net, 16, 64, scope='fire2')
        net = fire_module(net, 16, 64, scope='fire3')
        net = fire_module(net, 32, 128, scope='fire4')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool4')
        net = fire_module(net, 32, 128, scope='fire5')
        net = fire_module(net, 48, 192, scope='fire6')
        net = fire_module(net, 48, 192, scope='fire7')
        net = fire_module(net, 64, 256, scope='fire8')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
        net = fire_module(net, 64, 256, scope='fire9')
        net = slim.dropout(net, dropout_keep_prob) # ADD
        net = conv2d(net, num_classes, [1, 1], stride=1, scope='conv10')
        # net = avg_pool2d(net, [13, 13], stride=1, scope='avgpool10')
        net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avgpool10')
        # logits = tf.squeeze(net, [2], name='logits')
        net = tf.squeeze(net, [1, 2], name='logits')
        net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=False) # ADD
        return net, None



def _arg_scope(is_training, weight_decay, bn_decay):
    with arg_scope([conv2d],
                   weights_regularizer=l2_regularizer(weight_decay),
                   normalizer_fn=batch_norm,
                   normalizer_params={'is_training': is_training,
                                      'fused': True,
                                      'decay': bn_decay}):
        with arg_scope([conv2d, avg_pool2d, max_pool2d, batch_norm],
                       data_format='NHWC') as sc:
                return sc


'''
Network in Network: https://arxiv.org/abs/1312.4400
See Section 3.2 for global average pooling
'''


def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    args = {'num_classes': 1000,
     'weight_decay': weight_decay, 
     'batch_norm_decay':0.995, 
    'bottleneck_layer_size':bottleneck_layer_size, 
    'dropout_keep_prob':keep_probability}
    model = Squeezenet(args)
    with arg_scope([conv2d],
                   weights_regularizer=l2_regularizer(weight_decay),
                   normalizer_fn=batch_norm,
                   normalizer_params={'is_training': phase_train,
                                      'fused': True,
                                      'decay': 0.995}):
        with arg_scope([conv2d, avg_pool2d, max_pool2d, batch_norm],
                       data_format='NHWC') as sc:
            return model.build(images, is_training=phase_train)


    