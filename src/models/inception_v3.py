from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

trunc_normal = lambda std:tf.truncated_normal_initializer(0.0,std)
  
  # TODO，weight decay
def inference(images, keep_probability, phase_train=True, 
              bottleneck_layer_size=128, weight_decay=0.0, std=0.1, reuse=None):
    batch_norm_var_collection="moving_vars"
    batch_norm_params = {"decay":0.9997,"epsilon":0.001,"updates_collections":tf.GraphKeys.UPDATE_OPS,
                         "variables_collections":{"beta":None,"gamma":None,
                          "moving_mean":[batch_norm_var_collection],"moving_variance":[batch_norm_var_collection]}}
    
    with slim.arg_scope([slim.conv2d,slim.fully_connected],weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d],weights_initializer=tf.truncated_normal_initializer(stddev=std),
                            activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            return inceptionV3(images, is_training=phase_train,
                dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)

def inception_v3_base(inputs,scope=None):
    #保存关键节点
    end_points = {}
    with tf.variable_scope(scope,"InceptionV3",[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding="VALID"):
            net = slim.conv2d(inputs,32,[3,3],stride=2,scope="Conv2d_1a_3x3")
            net = slim.conv2d(net,32,[3,3],scope="Conv2d_2a_3x3")
            net = slim.conv2d(net,64,[3,3],padding="SAME",scope="Conv2d_2b_3x3")
            net = slim.max_pool2d(net,[3,3],stride=2,scope="MaxPool_3a_3x3")
            net = slim.conv2d(net,80,[1,1],scope="Conv2d_3b_1x1")
            net = slim.conv2d(net,192,[3,3],scope="Conv2d_4a_3x3")
            net = slim.max_pool2d(net,[3,3],stride=2,scope="MaxPool5a_3x3")
    #定义Inception module
    with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding="SAME"):
        with tf.variable_scope("Mixed_5b"):
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,48,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = slim.conv2d(branch_1,64,[5,5],scope="Conv2d_0b_5x5")
            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,96,[3,3],scope="Conv2d_0b_3x3")
                branch_2 = slim.conv2d(branch_2,96,[3,3],scope="Conv2d_0c_3x3")
            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,32,[1,1],scope="Conv2d_0b_1x1")
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        with tf.variable_scope("Mixed_5c"):
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,48,[1,1],scope="Conv2d_0b_1x1")
                branch_1 = slim.conv2d(branch_1,64,[5,5],scope="Conv_1_0c_5x5")
            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,96,[3,3],scope="Conv2d_0b_3x3")
                branch_2 = slim.conv2d(branch_2,96,[3,3],scope="Conv2d_0c_3x3")
            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,64,[1,1],scope="Conv2d_0b_1x1")
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        with tf.variable_scope("Mixed_5d"):
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,48,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = slim.conv2d(branch_1,64,[5,5],scope="Conv2d_0b_5x5")
            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,96,[3,3],scope="Conv2d_0b_3x3")
                branch_2 = slim.conv2d(branch_2,96,[3,3],scope="Conv2d_0c_3x3")
            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,64,[1,1],scope="Conv2d_0b_1x1")
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)

        with tf.variable_scope("Mixed_6a"):
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,384,[3,3],stride=2,padding="VALID",scope="Conv2d_1a_1x1")
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,64,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = slim.conv2d(branch_1,96,[3,3],scope="Conv2d_0b_3x3")
                branch_1 = slim.conv2d(branch_1,96,[3,3],stride=2,padding="VALID",scope="Conv2d_1a_1x1")
            with tf.variable_scope("Branch_2"):
                branch_2 = slim.max_pool2d(net,[3,3],stride=2,padding="VALID",scope="MaxPool_1a_3x3")
            net = tf.concat([branch_0,branch_1,branch_2],3)
        with tf.variable_scope("Mixed_6b"):
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,128,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = slim.conv2d(branch_1,128,[1,7],scope="Conv2d_0b_1x7")
                branch_1 = slim.conv2d(branch_1,192,[7,1],scope="Conv2d_0c_7x1")
            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,128,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,128,[7,1],scope="Conv2d_0b_7x1")
                branch_2 = slim.conv2d(branch_2,128,[1,7],scope="Conv2d_0c_1x7")
                branch_2 = slim.conv2d(branch_2,128,[7,1],scope="Conv2d_0d_7x1")
                branch_2 = slim.conv2d(branch_2,192,[1,7],scope="Conv2d_0e_1x7")
            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1x1")
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        with tf.variable_scope("Mixed_6c"):
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,160,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = slim.conv2d(branch_1,160,[1,7],scope="Conv2d_0b_1x7")
                branch_1 = slim.conv2d(branch_1,192,[7,1],scope="Conv2d_0c_7x1")
            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,160,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,160,[7,1],scope="Conv2d_0b_7x1")
                branch_2 = slim.conv2d(branch_2,160,[1,7],scope="Conv2d_0c_1x7")
                branch_2 = slim.conv2d(branch_2,160,[7,1],scope="Conv2d_0d_7x1")
                branch_2 = slim.conv2d(branch_2,192,[1,7],scope="Conv2d_0e_1x7")
            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1x1")
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        with tf.variable_scope("Mixed_6d"):
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,160,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = slim.conv2d(branch_1,160,[1,7],scope="Conv2d_0b_1x7")
                branch_1 = slim.conv2d(branch_1,192,[7,1],scope="Conv2d_0c_7x1")
            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,160,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,160,[7,1],scope="Conv2d_0b_7x1")
                branch_2 = slim.conv2d(branch_2,160,[1,7],scope="Conv2d_0c_1x7")
                branch_2 = slim.conv2d(branch_2,160,[7,1],scope="Conv2d_0d_7x1")
                branch_2 = slim.conv2d(branch_2,192,[1,7],scope="Conv2d_0e_1x7")
            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1x1")
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        with tf.variable_scope("Mixed_6e"):
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = slim.conv2d(branch_1,192,[1,7],scope="Conv2d_0b_1x7")
                branch_1 = slim.conv2d(branch_1,192,[7,1],scope="Conv2d_0c_7x1")
            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,192,[7,1],scope="Conv2d_0b_7x1")
                branch_2 = slim.conv2d(branch_2,192,[1,7],scope="Conv2d_0c_1x7")
                branch_2 = slim.conv2d(branch_2,192,[7,1],scope="Conv2d_0d_7x1")
                branch_2 = slim.conv2d(branch_2,192,[1,7],scope="Conv2d_0e_1x7")
            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1x1")
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        end_points["Mixed_6e"] = net

        with tf.variable_scope("Mixed_7a"):
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
                branch_0 = slim.conv2d(branch_0,320,[3,3],stride=2,padding="VALID",scope="Conv2d_1a_3x3")
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,192,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = slim.conv2d(branch_1,192,[1,7],scope="Conv2d_0b_1x7")
                branch_1 = slim.conv2d(branch_1,192,[7,1],scope="Conv2d_0b_7x1")
                branch_1 = slim.conv2d(branch_1,192,[3,3],stride=2,padding="VALID",scope="Conv2d_1a_3x3")
            with tf.variable_scope("Branch_2"):
                branch_2 = slim.max_pool2d(net,[3,3],stride=2,padding="VALID",scope="MaxPool_1a_3x3")
            net = tf.concat([branch_0,branch_1,branch_2],3)
        with tf.variable_scope("Mixed_7b"):
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,320,[1,1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,384,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = tf.concat([
                    slim.conv2d(branch_1,384,[1,3],scope="Conv2d_0b_1x3"),
                    slim.conv2d(branch_1,384,[3,1],scope="Conv2d_0b_3x1")
                ],3)
            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,448,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,384,[3,3],scope="Conv2d_0b_3x3")
                branch_2 = tf.concat([
                    slim.conv2d(branch_2,384,[1,3],scope="Conv2d_0c_1x3"),
                    slim.conv2d(branch_2,384,[3,1],scope="Conv2d_0d_3x1")
                ],3)
            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1x1")
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        with tf.variable_scope("Mixed_7c"):
            with tf.variable_scope("Branch_0"):
                branch_0 = slim.conv2d(net,320,[1,1],scope="Conv2d_0a_1x1")
            with tf.variable_scope("Branch_1"):
                branch_1 = slim.conv2d(net,384,[1,1],scope="Conv2d_0a_1x1")
                branch_1 = tf.concat([
                    slim.conv2d(branch_1,384,[1,3],scope="Conv2d_0b_1x3"),
                    slim.conv2d(branch_1,384,[3,1],scope="Conv2d_0c_3x1")
                ],3)
            with tf.variable_scope("Branch_2"):
                branch_2 = slim.conv2d(net,448,[1,1],scope="Conv2d_0a_1x1")
                branch_2 = slim.conv2d(branch_2,384,[3,3],scope="Conv2d_0b_3x3")
                branch_2 = tf.concat([
                    slim.conv2d(branch_2,384,[1,3],scope="Conv2d_0c_1x3"),
                    slim.conv2d(branch_2,384,[3,1],scope="Conv2d_0d_3x1")
                ],3)
            with tf.variable_scope("Branch_3"):
                branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3x3")
                branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1x1")
            net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        return net,end_points

def inceptionV3(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None, 
                        scope='InceptionV3'):
    end_points = {}
  
    with tf.variable_scope(scope, 'InceptionV3', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
            net,end_points = inception_v3_base(inputs,scope=scope)
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding="SAME"):
            aux_logits = end_points["Mixed_6e"]
            with tf.variable_scope("AuxLogits"):
                aux_logits = slim.avg_pool2d(aux_logits,[5,5],stride=3,padding="VALID",scope="AvgPool_1a_5x5")
                aux_logits = slim.conv2d(aux_logits,128,[1,1],scope="Conv2d_1b_1x1")
                kernel_size = _reduced_kernel_size_for_small_input(aux_logits, [5, 5])
                aux_logits = slim.conv2d(aux_logits,768,kernel_size,weights_initializer=trunc_normal(0.01),
                                         padding="VALID",scope='Conv2d_2a_{}x{}'.format(*kernel_size))
                aux_logits = slim.flatten(aux_logits)
                # aux_logits = slim.dropout(aux_logits, dropout_keep_prob, is_training=is_training,
                #                        scope='Dropout')
            with tf.variable_scope("Logits"):
                kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
                net = slim.avg_pool2d(net,kernel_size,padding="VALID",scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                net = slim.dropout(net,keep_prob=dropout_keep_prob,scope="Dropout_1b")
                net = slim.flatten(net)
                # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                #                        scope='Dropout')
            aux_logits = slim.fully_connected(aux_logits, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck2', reuse=False)
            net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck1', reuse=False)
            
    return net, aux_logits



def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.tf.contrib.slim.ops._two_element_tuple
  cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                        tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [
        min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])
    ]
  return kernel_size_out