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
model.path_lengths()
print('%5s|%13s|%13s|%14s|%13s|%6s|%6s|%6s|%6s|%6s|%6s' % ('idx','layer','input neurons','output neurons','layer masses', 'sp_ip', 'lp_ip','rw_ip', 'sp_op', 'lp_op', 'rw_op'))
for i in range(model.num_layers):
    print('%5d|%13s|%13d|%14d|%13d|%6d|%6d|%6f|%6d|%6d|%6f' % (i, model.ll[i], model.input_neurons[i], model.output_neurons[i], model.lm[i], model.delta_ip[0,i], model.delta_ip[1,i], model.delta_ip[2,i], model.delta_op[0,i], model.delta_op[1,i], model.delta_op[2,i]))

