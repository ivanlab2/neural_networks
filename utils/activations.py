import numpy as np

def sigm(x):#Функция сигмоиды
    return 1.0/(1.0 + np.exp(-x))

def dsigm(x):#Производная сигмоиды
    return np.exp(-x)/np.power(1+np.exp(-x),2)

def leakyrelu(x, alpha=0.01): 
    return (x >= 0) * x + (x < 0) * x * alpha

def dleakyrelu(output, alpha=0.01):
    return (output >= 0) + alpha*(output < 0)

def tanh(x):#Гиперболический тангенс
  return np.tanh(x)

def dtanh(x):#Производная гиперболического тангенса
  return 1 - np.square(tanh(x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=3)[:,:,:,None]

def relu(x):
    return (x >= 0) * x

def relu2deriv(output):
    return output >= 0

def softmax_last(x):#Функция softmax для выходного слоя
    return np.exp(x)/np.sum(np.exp(x),axis=1)[:,None]

def d_softmax(X):
    s=X.reshape(1,-1)
    return np.diagflat(s)-np.dot(s,s.T)

def d_softmax_cross_entropy(y_pred,y, n_classes=2): #Производная сразу и по функции потерь, и по последнему слою активации
    y_real=np.zeros([len(y),n_classes])
    for i in range(len(y)):
        y_real[i,y[i]]=1
    return y_pred-y_real

