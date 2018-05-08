"""VGG16 benchmark in Fluid"""
from __future__ import print_function

import sys
import time
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import functools

class VGG16(object):
    def __init__(self, data_set):
        #data_format is "NCHW"
        self.data_set_dict = {"cifar10": {"class_dim": 10, "data_shape" :[3, 32, 32]},
                              "flowers": {"class_dim": 102, "data_shape": [3, 224, 224]}}
        self.data_set = data_set
        self.class_dim = self.data_set_dict[self.data_set]["class_dim"]
        self.data_shape = self.data_set_dict[self.data_set]["data_shape"]

    def vgg16_bn_drop(self, input):
        def conv_block(input, num_filter, groups, dropouts):
            return fluid.nets.img_conv_group(
                input=input,
                pool_size=2,
                pool_stride=2,
                conv_num_filter=[num_filter] * groups,
                conv_filter_size=3,
                conv_act='relu',
                conv_with_batchnorm=True,
                conv_batchnorm_drop_rate=dropouts,
                pool_type='max')
        conv1 = conv_block(input, 64, 2, [0.3, 0])
        conv2 = conv_block(conv1, 128, 2, [0.4, 0])
        conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
        conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
        conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

        drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
        fc1 = fluid.layers.fc(input=drop, size=512, act=None)
        bn = fluid.layers.batch_norm(input=fc1, act='relu')
        drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
        fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
        return fc2

    def construct_vgg16_net(self, learning_rate):
        images = fluid.layers.data(name='pixel', shape=self.data_shape, dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        net = self.vgg16_bn_drop(images)
        predict = fluid.layers.fc(input=net, size=self.class_dim, act='softmax')
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        self.avg_cost = fluid.layers.mean(x=cost)
        self.accuracy = fluid.layers.accuracy(input=predict, label=label)

        self.inference_program = fluid.default_main_program().clone(for_test=True)

        optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
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
                if self.data_set == "cifar10" else paddle.dataset.flowers.train(),
                buf_size=5120),
            batch_size=batch_size)
        iters = 0
        for pass_id in range(num_passes):
            # train
            start_time = time.time()
            for batch_id, data in enumerate(train_reader()):
                img_data = np.array(map(lambda x: x[0].reshape(self.data_shape),
                                        data)).astype("float32")
                y_data = np.array(map(lambda x: x[1], data)).astype("int64")
                y_data = y_data.reshape([-1, 1])

                loss, acc = exe.run(self.train_program,
                                    feed={"pixel": img_data,
                                        "label": y_data},
                                    fetch_list=[self.avg_cost, self.accuracy])
                iters += 1
                print(
                    "Pass = %d, Iters = %d, Loss = %f, Accuracy = %f" %
                    (pass_id, iters, loss, acc))

if __name__ == "__main__":
    vgg16 = VGG16(data_set="cifar10")
    vgg16.construct_vgg16_net(learning_rate=1e-3)
    vgg16.train(batch_size=128, device="CPU", num_passes=50)

