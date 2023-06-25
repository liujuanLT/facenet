from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import os
import sys
import facenet
from six.moves import xrange  # @UnresolvedImport
sys.path.append('/home/jliu/codes/facenet/src')
from models import inception_resnet_v1
import numpy as np

# ref: https://github.com/LTyanghuang/ssd_resnet18_224_tf1.x_ok/blob/master/ckpt_ckpt2savedmodel_1.py

def get_tf_version():
    ver = tf.__version__.split('.')
    return int(ver[0]), int(ver[1])

def modify_avgpool_to_reducemean():
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    print(tensor_name_list)
    prev = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Block8/add:0")
    old = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool:0")
    new = tf.reduce_mean(prev, axis=[1,2], keepdims=True, name='InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool')
    comsus = old.consumers()
    for comsu in comsus:
        for i, inp in enumerate(comsu.inputs):
            if inp is old:
                comsu._update_input(i, new)
    return

def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    # Replace all the variables in the graph with constants of the same values
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names)
    return output_graph_def

def save_model(save_to_saved_model=True, saved_model_dir=None, save_to_pb=True, out_pb_file=None):
    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)

    image_shape = (160, 160)
    image_batch = tf.placeholder(shape=[1, *image_shape, 3], dtype=tf.float32, name='image_batch')
    reuse = None
    net = inception_resnet_v1
    prelogits, _ = net.inference(image_batch, 1.0, phase_train=False, bottleneck_layer_size=512, weight_decay=0.0)
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')            

    # restore
    ckpt_file = '/home/jliu/codes/facerecog_pubdata/models/download_weights/facenet_tf/20180402-114759/model-20180402-114759.ckpt-275'
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_file)

    modify_avgpool_to_reducemean()

    if save_to_saved_model:
        out_dir = '/home/jliu/codes/data/facenet_tf/facenet_saved_model'
        # out_dir = '/home/jliu/codes/data/facerecog/facenet_saved_model'
        tf.saved_model.simple_save(isess, out_dir, inputs={'image_batch': image_batch}, 
            outputs={'embeddings': embeddings})
        print(f'finished saved model to {out_dir}')

    if save_to_pb:
        input_graph_def = isess.graph.as_graph_def()
        output_graph_def = freeze_graph_def(isess, input_graph_def, ['embeddings'])
        # out_pb_file = '/home/jliu/codes/data/facenet_tf/20180402-114759_modify.pb'
        out_pb_file = '/home/jliu/codes/data/facerecog/20180402-114759_modify.pb'
        with tf.gfile.GFile(out_pb_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), out_pb_file))

def inference_pb(model_file=None):
    
    with tf.gfile.FastGFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    embed = tf.import_graph_def(graph_def, return_elements=['embeddings:0'])
 
    image_shape = (160, 160)
    # image_path = '/home/swmkt/huangyang/dagger_benchmark/SDKExamples/panda.jpg'
    image_path = '/home/jliu/codes/facerecog_pubdata/datasets/lfw/lfw-112X96/Abel_Pacheco/Abel_Pacheco_0001.jpg'
    major, minor = get_tf_version()
    assert major == 1
    assert minor == 7
    if minor <= 7:   
        raw_image = tf.read_file(image_path, 'r')
        image_data = tf.image.decode_jpeg(raw_image, 3)
        image_data = tf.image.resize_image_with_crop_or_pad(image_data, image_shape[0], image_shape[1])
        image_data = tf.image.per_image_standardization(image_data)
        image_data.set_shape(image_shape + (3,))
    else:
        raw_image = tf.io.read_file(image_path, 'r')
        image_data = tf.image.decode_jpeg(raw_image, channels=3, acceptable_fraction=3)
        # image_data = tf.cast(image_data, tf.float32) / 128. - 1
        # image_data = tf.image.resize(image_data, image_shape)
        image_data = tf.image.resize_image_with_crop_or_pad(image_data, image_shape[0], image_shape[1])
        image_data = tf.image.per_image_standardization(image_data)

    image_input = tf.expand_dims(image_data, 0)
    input_tensor = tf.get_default_graph().get_tensor_by_name('import/image_batch:0')
    output_tensor = tf.get_default_graph().get_tensor_by_name('import/embeddings:0')

    with tf.Session() as sess:
        image_input = sess.run(image_input) # tensor to list
        embed_res = sess.run(output_tensor, feed_dict={input_tensor: image_input})
        print(f'embeddings:\n{embed_res}')
    return

if __name__ == '__main__':
    save_model()
    # inference_pb(model_file='/home/jliu/codes/data/facenet_tf/20180402-114759_modify2.pb')