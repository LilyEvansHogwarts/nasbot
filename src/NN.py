import autograd.numpy as np

# in the ll and lu vector, input and output will always be the first and last item
# softmax layers stay next to the output layer
# in vector lu, the categories of each layer will be assigned a number, which is the same as matrix M
# path is a 2-dimensional vector, the element for the input is [], since there is no input layer for the input
# for j in self.path1[i]: j < i
# for j in self.path2[i]: j > i

class NN:
    def __init__(self, ll, lu, path):
        self.ll = np.copy(ll)
        self.lu = np.copy(lu)
        self.path1 = path
        self.num_layers = self.lu.size
        self.zeta = 0.1

    def ispool(self, i):
        return self.ll[i][4:] == 'pool'

    def calc_path2(self):
        path2 = []
        for i in range(self.num_layers):
            path2.append([])
        for i in range(self.num_layers):
            for j in self.path1[i]:
                path2[j].append(i)
        self.path2 = path2
    
    # layer_masses
    def calc_lm(self):
        self.lm = np.zeros((self.num_layers))
        self.input_neurons = np.zeros((self.num_layers))
        self.output_neurons = np.zeros((self.num_layers))
        self.output_neurons[0] = self.lu[0]
        for i in range(1, self.num_layers):
            if self.ll[i] == 'ip' or self.ll[i] == 'op' or self.ll[i] == 'softmax':
                continue
            else:
                self.input_neurons[i] = np.array([self.output_neurons[j] for j in self.path1[i]]).sum()
                
                self.output_neurons[i] = self.input_neurons[i] if self.ispool(i) else self.lu[i]
                self.lm[i] = self.input_neurons[i] if self.ispool(i) else self.input_neurons[i]*self.output_neurons[i]
        # ip and op
        tmp = self.zeta * self.lm.sum()
        self.lm[0] = tmp
        self.lm[-1] = tmp
        # softmax layers
        softmax_idx = np.arange(self.num_layers)[self.ll == 'softmax']
        self.lm[softmax_idx] = tmp/(self.ll == 'softmax').sum()
    
    # path lengths
    def path_lengths(self):
        self.delta_ip = np.zeros((3, self.num_layers))
        self.delta_op = np.zeros((3, self.num_layers))
        for i in range(self.num_layers):
            self.delta_ip[0,i] = np.array([self.delta_ip[0,j] for j in self.path1[i]]).min() + 1
            self.delta_ip[1,i] = np.array([self.delta_ip[1,j] for j in self.path1[i]]).max() + 1
        for i in np.arange(self.num_layers,0,-1)-1:
            self.delta_op[0,i] = np.array([self.delta_ip[0,j] for j in self.path2[i]]).min() + 1
            self.delta_op[1,i] = np.array([self.delta_ip[1,j] for j in self.path2[i]]).min() + 1





