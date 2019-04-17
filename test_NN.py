import autograd.numpy as np
from src.NN import NN
import toml
import sys

argv = sys.argv[1:]
net = toml.load(argv[0])

label = ['ip', 'op', 'softmax', 'conv3', 'conv5', 'conv7', 'max-pool', 'avg-pool', 'fc']


model = NN(net['ll'], net['lu'], net['path'])
model.calc_path2()
print('path2', model.path2)
model.calc_lm()
print('input_neurons', model.input_neurons)
print('output_neurons', model.output_neurons)
print('layer masses', model.lm)

