import autograd.numpy as np
from src.NN import NN
from src.Distance import Distance
import sys
import toml

argv = sys.argv[1:]
conf1 = toml.load(argv[0])
conf2 = toml.load(argv[1])

nn1 = NN(conf1['ll'], conf1['lu'], conf1['path'])
nn2 = NN(conf2['ll'], conf2['lu'], conf2['path'])
labels = {'ip':0, 'conv3':1, 'conv5':2, 'conv7':3, 'max-pool':4, 'avg-pool':5, 'fc':6, 'softmax':7, 'op':8}
cost_matrix = np.ones((len(labels), len(labels))) * 10000
cost_matrix[range(len(labels)), range(len(labels))] = 0
cost_matrix[labels['conv3'], labels['conv5']] = cost_matrix[labels['conv5'], labels['conv3']] = 0.2
cost_matrix[labels['conv5'], labels['conv7']] = cost_matrix[labels['conv7'], labels['conv5']] = 0.3
cost_matrix[labels['conv3'], labels['res3']] = cost_matrix[labels['res3'], labels['conv3']] = 0.1
cost_matrix[labels['conv5'], labels['res5']] = cost_matrix[labels['res5'], labels['conv5']] = 0.1
cost_matrix[labels['conv7'], labels['res7']] = cost_matrix[labels['res7'], labels['conv7']] = 0.1
cost_matrix[labels['max-pool'], labels['avg-pool']] = cost_matrix[labels['avg-pool'], labels['max-pool']] = 0.25

model = Distance(nn1, nn2, cost_matrix, labels)
print(model.M)
print(model.C)
Z = model.rand_Z()
model.train()
print(model.Z)
print(model.Label_mismatch(model.Z) + model.Non_assignment(model.Z) + model.Structure(model.Z))
