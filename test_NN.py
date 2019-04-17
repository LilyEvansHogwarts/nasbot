import autograd.numpy as np
from src.NN import NN
import toml
import sys

argv = sys.argv[1:]
net = toml.load(argv[0])

label = ['ip', 'op', 'softmax', 'conv3', 'conv5', 'conv7', 'max-pool', 'avg-pool', 'fc']


model = NN(net['ll'], net['lu'], net['path'])
model.calc_path2()
model.calc_lm()
print('path2', model.path2)
print('%5s|%13s|%13s|%14s|%13s' % ('idx','layer','input neurons','output neurons','layer masses'))
for i in range(model.num_layers):
    print('%5d|%13s|%13d|%14d|%13d' % (i, model.ll[i], model.input_neurons[i], model.output_neurons[i], model.lm[i]))


