import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import fmin_l_bfgs_b
import traceback

class Distance:
    def __init__(self, nn1, nn2, cost_matrix, labels):
        self.nn1 = nn1
        self.nn2 = nn2
        self.cost_matrix = cost_matrix
        self.labels = labels
        self.calc_M()
        self.calc_C()

    def calc_M(self):
        self.M = np.zeros((self.nn1.num_layers, self.nn2.num_layers))
        tmp1 = [self.cost_matrix[self.labels[i]] for i in self.nn1.ll]
        tmp2 = [self.labels[j] for j in self.nn2.ll]
        self.M = np.array([tmp1[i][tmp2] for i in range(self.nn1.num_layers)])

    def calc_C(self):
        self.C = np.zeros((self.nn1.num_layers, self.nn2.num_layers))
        for i in range(self.nn1.num_layers):
            for j in range(self.nn2.num_layers):
                self.C[i,j] = np.abs(self.nn1.delta_ip[:,i] - self.nn2.delta_ip[:,j]).sum() + np.abs(self.nn1.delta_op[:,i] - self.nn2.delta_op[:,j]).sum()
        self.C = self.C/6.0
                

    def rand_Z(self):
        Z = np.random.randn(self.nn1.num_layers, self.nn2.num_layers)
        Z[self.M > 1] = 0
        return Z

    def Label_mismatch(self, Z):
        return (self.M * Z).sum()

    def Non_assignment(self, Z):
        return (self.nn1.lm - Z.sum(axis=1)).sum() + (self.nn2.lm - Z.sum(axis=0)).sum()

    def Structure(self, Z):
        return (self.C * Z).sum()

    def train(self):
        Z0 = self.rand_Z()
        self.Z = np.copy(Z0)
        self.best_loss = np.inf
        self.tmp_loss = np.inf

        def loss(Z):
            Z = Z.reshape((self.nn1.num_layers,-1))
            nlz = (self.M * Z + self.C * Z).sum() + (self.nn1.lm - Z.sum(axis=1)).sum() + (self.nn2.lm - Z.sum(axis=0)).sum()
            # nlz = self.Label_mismatch(Z) + self.Non_assignment(Z) + self.Structure(Z)
            self.tmp_loss = nlz
            return nlz

        def callback(Z):
            if self.tmp_loss < self.best_loss:
                self.best_loss = self.tmp_loss
                self.Z = np.copy(Z)

        gloss = value_and_grad(loss)

        bounds = []
        for i in range(self.nn1.num_layers):
            for j in range(self.nn2.num_layers):
                bounds.append([0, np.min([self.nn1.lm[i], self.nn2.lm[j]])])

        try:
            fmin_l_bfgs_b(gloss, Z0, bounds=bounds, maxiter=1000, m=100, iprint=0, callback=callback)
        except:
            print('Failed')
            print(traceback.format_exc())

        self.Z = self.Z.reshape(self.nn1.num_layers, self.nn2.num_layers)


