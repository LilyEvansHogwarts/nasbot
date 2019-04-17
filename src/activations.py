import autograd.numpy as np

SQRT_2 = np.sqrt(2)
PI = np.pi
PI_2 = 2*np.pi
SQRT_2PI = np.sqrt(2*np.pi)

def get_act(name):
    if name == 'cos':
        return cos
    elif name == 'relu':
        return relu
    elif name == 'tanh':
        return tanh
    elif name == 'sigmoid':
        return sigmoid
    else:
        print('There is no activation function named:', name)

def relu(x):
    return np.maximum(0, x)

def cos(x):
    return np.cos(x)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def pdf(x):
    return np.exp(-x**2 / 2)/np.sqrt(2*PI)

def cdf(x):
    return 0.5 + erf(x/SQRT_2)/2

def erf(x):
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
                            
    # Save the sign of x
    sign = np.sign(x)
    x = np.abs(x)
                                                        
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x**2)
                                                                    
    return sign*y


