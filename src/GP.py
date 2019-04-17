import autograd.numpy as np
import traceback
import sys
from autograd import value_and_grad
from scipy.optimize import fmin_l_bfgs_b

def chol_inv(L, y):
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

class GP:
    def __init__(self, train_x, train_y, bfgs_iter=1000, debug=False):
        self.train_x = np.copy(train_x)
        self.train_y = np.copy(train_y).reshape(-1)
        self.mean = self.train_y.mean()
        self.std = self.train_y.std()
        self.train_y = (self.train_y - self.mean)/(0.000001 + self.std)
        self.bfgs_iter = bfgs_iter
        self.debug = debug
        self.dim, self.num_train = self.train_x.shape

    def rand_theta(self, scale):
        theta = scale * np.random.randn(2+self.dim)
        theta[0] = np.log(np.std(self.train_y))-30
        theta[1] = np.log(np.std(self.train_y))
        for i in range(self.dim):
            theta[2+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        return theta

    def kernel(self, x, xp, hyp):
        sf2 = np.exp(hyp[0])
        lengthscale = np.exp(hyp[1:])+0.000001
        x = (x.T/lengthscale).T
        xp = (xp.T/lengthscale).T
        diff = 2*np.dot(x.T,xp) - (x**2).sum(axis=0)[:,None] - (xp**2).sum(axis=0)
        return sf2*np.exp(0.5*diff)

    def neg_likelihood(self, theta):
        sn2 = np.exp(theta[0])
        hyp = theta[1:]
        K = self.kernel(self.train_x, self.train_x, hyp) + sn2 * np.eye(self.num_train)
        L = np.linalg.cholesky(K)
        
        logDetK = 2*np.sum(np.log(np.diag(L)))
        datafit = np.dot(self.train_y, chol_inv(L, self.train_y))
        neg_likelihood = 0.5*(datafit + logDetK)
        return neg_likelihood

    def train(self, scale=0.5, theta=None):
        if theta is None:
            theta0 = self.rand_theta(scale)
        else:
            theta0 = np.copy(theta)
        self.theta = np.copy(theta0)
        self.best_loss = np.inf
        self.tmp_loss = np.inf

        def loss(theta):
            nlz = self.neg_likelihood(theta)
            self.tmp_loss = nlz
            return nlz

        def callback(theta):
            if self.tmp_loss < self.best_loss:
                self.best_loss = self.tmp_loss
                self.theta = np.copy(theta)

        gloss = value_and_grad(loss)
         
        try:
            fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=100, iprint=self.debug, callback=callback)
        except np.linalg.LinAlgError:
            print('GP model. Increase noise term and re-optimize.')
            theta0 = np.copy(self.theta)
            theta0[0] += np.log(10)
            try:
                fmin_l_bfgs_b(gloss, theta0, maxiter=self.bfgs_iter, m=10, iprint=self.debug, callback=callback)
            except:
                print('GP model. Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            if self.debug:
                print('GP model. Exception caught, L-BFGS early stopping...')
                print(traceback.format_exc())

        if np.isinf(self.best_loss) or np.isnan(self.best_loss):
            print('GP model. Fail to build GP model.')
          
        sn2 = np.exp(self.theta[0])
        K = self.kernel(self.train_x, self.train_x, self.theta[1:]) + sn2 * np.eye(self.num_train)
        self.L = np.linalg.cholesky(K)
        self.alpha = chol_inv(self.L, self.train_y)
        print('GP model. Finish training process')

    def predict(self, test_x):
        sn2 = np.exp(self.theta[0])
        tmp = self.kernel(test_x, self.train_x, self.theta[1:])
        py = np.dot(tmp, self.alpha) * self.std + self.mean
        ps2 = sn2 + np.exp(self.theta[1]) - (tmp * chol_inv(self.L, tmp.T).T).sum(axis=1)
        ps2 = np.abs(ps2) * (self.std)**2
        return py, ps2

    def get_theta(self):
        return self.theta
