# coding=utf-8
import time
import sys
import json
import os
import time
import pickle as pkl

import numpy as np
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import step_stats_pb2
import google.protobuf.text_format as pbtf
from tensorflow.python.distribute.distribution_strategy_context import experimental_set_strategy

sys.path.append('../')
sys.path.append('./modeltransformer/')
sys.path.append('./bert/')

from profiler import Profiler
from profiler import NcclProfiler
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver
import traceback

from tensorflow.contrib.slim.nets import vgg, resnet_v2, inception
from tf_models.models.research.slim.nets.mobilenet import mobilenet_v2
from tf_models.models.research.slim.nets.nasnet import nasnet

import modeltransformer.transformer as transf
from modeltransformer.data import DatasetManager

from bert.runsquad import new_model_fn_builder
import modeling


def get_config_dict():
    config_dict = dict()
    if os.path.exists('config.txt'):
        with open('config.txt', 'r') as f:
            config_dict = json.load(f)

    return config_dict

def setup_server(workers):
    #workers = ['10.28.1.26:3901', '10.28.1.25:3901','10.28.1.24:3901','10.28.1.17:3901','10.28.1.16:3901']
    #workers = ['10.28.1.26:3901','10.28.1.17:3901','10.28.1.16:3901']
    #os.environ['TF_CONFIG'] = '{ 'cluster': { 'worker': ['10.28.1.26:3901','10.28.1.17:3901','10.28.1.16:3901']  }, 'task': {'type': 'worker', 'index': 0} }'

    clus = dict()
    clus['cluster'] = {'worker': workers}
    clus['task'] = {'type': 'worker', 'index': 0}
    os.environ['TF_CONFIG'] = json.dumps(clus)

    #setup_workers(workers, 'grpc')

    resolver = TFConfigClusterResolver()
    cluster = resolver.cluster_spec()
    '''
    dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)
    config = dist.update_config_proto(tf.ConfigProto())
    config.ClearField('device_filters')
    config.allow_soft_placement = True  # log_device_placement=True)
    config.gpu_options.allow_growth = True
    '''
    config = tf.ConfigProto()

    if os.path.exists('dist_config.pbtxt'):
        print('dist_config.pbtxt exists.')
        with open('dist_config.pbtxt', 'r') as f:
            txt = f.read()
        pbtf.Parse(txt, config)

    server = tf.distribute.Server(cluster, job_name='worker', task_index=0,
                                    protocol='grpc', config=config)

    return server


#def model_fn(model_name, batch_size, opt):
def model_fn(model_name, batch_size):
    if model_name=='vgg19':
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size,1000))
        output, _ = vgg.vgg_19(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)

    elif model_name=='resnet200':
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size,1,1, 1000))
        output, _ = resnet_v2.resnet_v2_200(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)

    elif model_name=='resnet101':
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size,1,1, 1000))
        output, _ = resnet_v2.resnet_v2_101(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)

    elif model_name=='resnet152':
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size,1,1, 1000))
        output, _ = resnet_v2.resnet_v2_152(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)

    elif model_name=='nasnet_cifar':
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size,1000))
        output, _ = nasnet.build_nasnet_cifar(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)

    elif model_name=='mobile_net':
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size,1000))
        output, _ = mobilenet_v2.mobilenet(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)

    elif model_name=='inceptionv3':
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size, 1000))
        output, _ = inception.inception_v3(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)

    elif model_name=='transformer':
        dm = DatasetManager('wmt14')
        dm.maybe_download_data_files()
        dm.load_vocab()
        transformer = transf.Transformer(
            num_heads=8,
            d_model=512,
            d_ff=2048,
            model_name=model_name,
            tf_sess_config=dict(allow_soft_placement=True)
        )
        train_params = dict(
            learning_rate=1e-4,
            batch_size=batch_size,
            seq_len=10,
            max_steps=300000,
        )
        transformer.build_model('wmt14', dm.source_id2word, dm.target_id2word, 0,**train_params)
        loss = transformer._loss

    elif model_name=='bert':
        #bert_config = modeling.BertConfig.from_json_file('bert/bert_large/bert_config.json')
        bert_large_config_path = 'bert/pre-trained/large/cased_L-24_H-1024_A-16/bert_config.json'
        bert_config = modeling.BertConfig.from_json_file(bert_large_config_path)
        model = new_model_fn_builder(bert_config)
        features = {}
        features['input_ids']= tf.cast(100*tf.placeholder(tf.float32,shape=(batch_size,128)),tf.int32)
        features['input_mask'] = tf.cast(100*tf.placeholder(tf.float32,shape=(batch_size,128)),tf.int32)
        features['segment_ids']=tf.cast(100*tf.placeholder(tf.float32,shape=(batch_size,128)),tf.int32)
        features['start_positions'] = tf.cast(100*tf.placeholder(tf.float32,shape=(batch_size,)),tf.int32)
        features['end_positions'] =tf.cast(100*tf.placeholder(tf.float32,shape=(batch_size,)),tf.int32)
        loss = model(features)

    elif model_name == 'small':
        slim = tf.contrib.slim
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size, 1000))
        v= tf.get_variable(name='large_variable',shape=(3000,224, 224, 3),trainable=True)
        x = tf.slice(v,[0,0,0,0],tf.shape(x),name='large_slice')
        net = slim.max_pool2d(x, [2, 2], 2)
        net = slim.conv2d(net, 128, [5, 5],trainable=False)
        net = slim.max_pool2d(net, [2, 2], 2)
        net = slim.conv2d(net, 128, [5, 5],trainable=False)
        net = slim.max_pool2d(net, [2, 2], 2)
        net = slim.conv2d(net, 128, [5, 5],trainable=False)
        net = slim.max_pool2d(net, [2, 2], 2)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024, activation_fn=tf.nn.sigmoid,trainable=False)
        net = slim.fully_connected(net, 1000, activation_fn=None,trainable=False)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=net)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.2,
                            beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(
                                                        tf.reduce_sum(loss))
    # TODO: Make lr, beta, epsilon value of parameter
    """
    if opt == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=0.2,
                            beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(
                                                        tf.reduce_sum(loss))
    elif opt == 'GradientDescent':
        optimizer = tf.train.GradientDescentOptimizer(
                                learning_rate=0.2).minimize(tf.reduce_sum(loss))
    """
    return optimizer


def generate_edge_file(null_gdef,folder):
    with open(folder+'graph.pbtxt','w') as f:
        f.write(pbtf.MessageToString(null_gdef))
    name_list = [nodedef.name for nodedef in null_gdef.node]
    item_list=[]
    for i, nodedef in enumerate(null_gdef.node):
        for j, input in enumerate(nodedef.input):
            if ':' in input:
                index = int(input.split(':')[1])
                input = input.split(':')[0]
            else:
                index = 0

            if input[0]=='^':
                input_node_idx = name_list.index(input[1:])
                output_node_idx = i
                item_list.append('{} {} {}'.format(input[1:], nodedef.name, 1))
            else:
                input_node_idx = name_list.index(input)
                #output_node_idx = i
                input_nodedef = null_gdef.node[input_node_idx]
                #output_shape = input_nodedef.attr['_output_shapes'].list.shape[index]
                #size = 1
                #for dim in output_shape.dim:
                #    size*=dim.size
                item_list.append('{} {} {}'.format(input_nodedef.name,nodedef.name,1))
    with open(folder+'edgelist.txt','w') as f:
        item_list = ['\n'+item if i!=0 else item for i,item in enumerate(item_list)]
        f.writelines(item_list)


def generate_nccl_model(devices, server):
    model = NcclProfiler(devices, server.target).profile()
    with open('data/nccl_model.pkl', 'wb') as f:
        pkl.dump(model, f)


def generate_feature_file(index, model_name, server, sinks, devices):
    # TODO: refactor config_dict['inputs']
    folder = 'data/graph'+ str(index + 1) + '/'
    os.makedirs(folder, exist_ok=True)
    if model_name == 'transformer':
        batch_size = 288
    elif model_name == 'bert':
        batch_size = 3
    else:
        batch_size = 24
    final_dict = dict()
    opt = model_fn(model_name, None)
    init = tf.global_variables_initializer()
    null_gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
    with open(folder + 'null_graph.pbtxt', 'w') as f:
        f.write(pbtf.MessageToString(null_gdef))
    tf.reset_default_graph()

    generate_edge_file(null_gdef, folder)
    if os.path.exists('op_type_dict.json'):
        with open('op_type_dict.json', 'r') as f:
            op_type_dict=json.load(f)
    else:
        op_type_dict = dict()
    if model_name == 'bert':
        replica_num = [1, 1, 2, 2, 3, 3, 3]
    else:
        replica_num = [1, 2, 3, 4, 6, 8, 12]
    item_list=[]
    times_dict=dict()
    for replica_times in range(len(replica_num)):
        tf.reset_default_graph()
        run_metadata = None
            #opt = model_fn(models[index],batch_size/replica_num[replica_times])
            #init = tf.global_variables_initializer()
            #gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
        profilers = []
        for _ in range(5): # gc: Why 5?
            even_batch_size = int(batch_size/replica_num[replica_times])
            profiler = Profiler(null_gdef, even_batch_size, server.target, sinks)
            profilers.append(profiler)
        for i, nodedef in enumerate(null_gdef.node):
            times = times_dict.get(nodedef.name, '')
            if op_type_dict.get(nodedef.op, -1) == -1:
                op_type_dict[nodedef.op] = len(op_type_dict.keys())
            print('[gc] devices:', devices)
            for j in range(len(devices)):
                print('device j:', j)
                time_list = []
                for k in range(5):
                    try:
                        print('[gc] nodedef.name:', nodedef.name)
                        print('[gc] devices[j]:', devices[j])
                        print('[gc] run_meatdata:', run_metadata)
                        time = profilers[k].profile(nodedef.name,
                                                    devices[j], run_metadata)
                    except Exception as ex:
                        print(sys.stderr, 'profile error: ', ex)
                        print(nodedef)
                        traceback.print_exc()
                        time = 0
                    time_list.append(time)
                new_time = min(time_list)
                item = final_dict.get(
                        (nodedef.name,replica_num[replica_times]), None)
                if item == None:
                    final_dict[(nodedef.name, replica_num[replica_times])] = list()
                    item = final_dict[(nodedef.name, replica_num[replica_times])]
                item.append(new_time)
                times += str(new_time) + ' '
            times_dict[nodedef.name] = times
    name_list = [nodedef.name for nodedef in null_gdef.node]
    for i, nodedef in enumerate(null_gdef.node):
        size = 0
        for j, input in enumerate(nodedef.input):
            if ':' in input:
                index = int(input.split(':')[1])
                input = input.split(':')[0]
            else:
                index = 0

            if input[0] == '^':
                continue
            else:
                input_node_idx = name_list.index(input)
                #output_node_idx = i
                input_nodedef = null_gdef.node[input_node_idx]
                output_shape = input_nodedef.attr['_output_shapes'].list.shape[index]
                local_size = 1
                for dim in output_shape.dim:
                    local_size *= dim.size
                size += local_size
        times = times_dict[nodedef.name]
        item_list.append('{} {} {}{} {}'.format(nodedef.name,
                                op_type_dict[nodedef.op], times,size,batch_size))
    for i, nodedef in enumerate(null_gdef.node):
        if nodedef.name not in name_list:
            item_list.append('{} {} {}{} {}'.format(nodedef.name,
                                op_type_dict[nodedef.op], 0, 0, batch_size))

    with open(folder + 'docs.txt','w') as f:
        item_list = ['\n' + item if i != 0 else item for i, item in enumerate(item_list)]
        f.writelines(item_list)
    with open('op_type_dict.json', 'w') as f:
        json.dump(op_type_dict,f)
    with open(folder+'cost.pkl', 'wb') as f:
        pkl.dump(final_dict,f)


def main():
    config_dict = get_config_dict()
    devices = config_dict.get('devices',
                            ['/job:worker/replica:0/task:0/device:GPU:0'])
    print('devices', devices)
    workers = config_dict.get('workers', ['127.0.0.1:30000'])
    print('workers', workers)
    sinks = config_dict.get('sinks')

    server = setup_server(workers)
    sinks = ['Adam'] # gc: Why only Adam..? --> go and see model_fn()
    models = [
       'vgg19', 'resnet200', 'mobile_net', 'resnet101', 'resnet152', 'nasnet_cifar',
       'inceptionv3', 'transformer', 'bert', 'small',
    ]
    models = [
       'resnet101', 'resnet152', 'inceptionv3', 'transformer', 'bert', 'small',
    ]
    #models = ['vgg19'] # TODO: models will be configured by config.txt

    #for idx, model_name in enumerate(models):
    #    tf.reset_default_graph()
    #    generate_feature_file(idx, model_name, server, sinks, devices)
    #"""
    idx = 4
    model_name = 'bert'
    tf.reset_default_graph()
    generate_feature_file(idx, model_name, server, sinks, devices)
    #"""
    """
    models = ['mobile_net', 'nasnet_cifar']
    for idx, model_name in enumerate(models):
        tf.reset_default_graph()
        if idx == 0:
            idx = 5
        else:
            idx = 6
        generate_feature_file(idx, model_name, server, sinks, devices)
    """
    #generate_nccl_model(devices, server)


if __name__ == '__main__':
    main()
