from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import numpy as np
import time

import cProfile, pstats, StringIO

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


def conv_bn_layer(input, ch_out, filter_size, stride, padding, act='relu'):
    conv1 = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv1, act=act)


def shortcut(input, ch_out, stride, data_format='NCHW'):
    ch_in = input.shape[1] if data_format == 'NCHW' else input.shape[-1]
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input


def basicblock(input, ch_out, stride, data_format='NCHW'):
    short = shortcut(input, ch_out, stride, data_format)
    conv1 = conv_bn_layer(input, ch_out, 3, stride, 1)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1, act=None)
    return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def bottleneck(input, ch_out, stride, data_format='NCHW'):
    short = shortcut(input, ch_out * 4, stride, data_format)
    conv1 = conv_bn_layer(input, ch_out, 1, stride, 0)
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1)
    conv3 = conv_bn_layer(conv2, ch_out * 4, 1, 1, 0, act=None)
    return fluid.layers.elementwise_add(x=short, y=conv3, act='relu')


def layer_warp(block_func, count, input, ch_out, stride, data_format='NCHW'):
    res_out = block_func(input, ch_out, stride, data_format)
    for i in range(1, count):
        res_out = block_func(res_out, ch_out, 1, data_format)
    return res_out


def resnet_imagenet(input, class_dim, depth=50, data_format='NCHW'):

    cfg = {
        18: ([2, 2, 2, 1], basicblock),
        34: ([3, 4, 6, 3], basicblock),
        50: ([3, 4, 6, 3], bottleneck),
        101: ([3, 4, 23, 3], bottleneck),
        152: ([3, 8, 36, 3], bottleneck)
    }
    stages, block_func = cfg[depth]
    conv1 = conv_bn_layer(input, ch_out=64, filter_size=7, stride=2, padding=3)
    pool1 = fluid.layers.pool2d(
        input=conv1, pool_type='avg', pool_size=3, pool_stride=2)
    res1 = layer_warp(block_func, stages[0], pool1, 64, 1, data_format)
    res2 = layer_warp(block_func, stages[1], res1, 128, 2, data_format)
    res3 = layer_warp(block_func, stages[2], res2, 256, 2, data_format)
    res4 = layer_warp(block_func, stages[3], res3, 512, 2, data_format)
    pool2 = fluid.layers.pool2d(
        input=res4,
        pool_size=7,
        pool_type='avg',
        pool_stride=1,
        global_pooling=True)
    out = fluid.layers.fc(input=pool2, size=class_dim, act='softmax')
    return out

def resnet_cifar10(input, class_dim, depth=32, data_format='NCHW'):
    assert (depth - 2) % 6 == 0

    n = (depth - 2) // 6

    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, n, conv1, 16, 1, data_format)
    res2 = layer_warp(basicblock, n, res1, 32, 2, data_format)
    res3 = layer_warp(basicblock, n, res2, 64, 2, data_format)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    out = fluid.layers.fc(input=pool, size=class_dim, act='softmax')
    return out


class ResNet(object):
    def __init__(self, data_set):
        #data_format is "NCHW"
        self.data_set_dict = {
            "cifar10": {
                "class_dim": 10,
                "data_shape": [3, 32, 32]
            },
            "flowers": {
                "class_dim": 102,
                "data_shape": [3, 224, 224]
            }
        }
        self.model_dict = {
        "cifar10": resnet_cifar10,
        "flowers": resnet_imagenet
        }
        self.data_set = data_set
        self.class_dim = self.data_set_dict[self.data_set]["class_dim"]
        self.data_shape = self.data_set_dict[self.data_set]["data_shape"]

    def construct_resnet(self, depth, learning_rate, momentum):
        input = fluid.layers.data(name='data', shape=self.data_shape, dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        predict = self.model_dict[self.data_set](input, self.class_dim, depth)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        self.avg_cost = fluid.layers.mean(x=cost)
        
        self.accuracy = fluid.layers.accuracy(input=predict, label=label)
        # inference program
        self.inference_program = fluid.default_main_program().clone(for_test=True)

        optimizer = fluid.optimizer.Momentum(learning_rate=learning_rate, momentum=momentum)
        opts = optimizer.minimize(self.avg_cost)
        fluid.memory_optimize(fluid.default_main_program())
        self.train_program = fluid.default_main_program().clone()

    def train(self, batch_size, device, num_passes):
        place = core.CPUPlace() if device == 'CPU' else core.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.cifar.train10()
                if self.data_set == 'cifar10' else paddle.dataset.flowers.train(),
                buf_size=5120),
            batch_size=batch_size)

        iters = 0
        for pass_id in range(num_passes):
            pass_duration = 0.0
            for batch_id, data in enumerate(train_reader()):
                batch_start = time.time()
                image = np.array(map(lambda x: x[0].reshape(self.data_shape),
                                     data)).astype('float32')
                label = np.array(map(lambda x: x[1], data)).astype('int64')
                label = label.reshape([-1, 1])
                loss, acc = exe.run(self.train_program,
                                feed={'data': image,
                                      'label': label},
                                fetch_list=[self.avg_cost, self.accuracy])
                print("Pass: %d, Iter: %d, loss: %s, acc: %s" %
                      (pass_id, iters, str(loss), str(acc)))
                iters += 1

if __name__ == '__main__':
    resnet50 = ResNet(data_set="cifar10")
    resnet50.construct_resnet(depth=50, learning_rate=0.01, momentum=0.9)
    resnet50.train(batch_size=32, device="CPU", num_passes=50)
