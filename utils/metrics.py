import numpy as np
import matplotlib.pyplot as plt

def accuracy(y_pred,y):
    return np.sum((y_pred == y))/y.shape[0]

def precision(y_pred,y):
    tp=np.sum((y_pred == y)&(y_pred==1))
    fp=np.sum((y_pred != y)&(y_pred==1))
    return tp/(tp+fp)

def recall(y_pred,y):
    tp=np.sum((y_pred == y)&(y_pred==1))
    fn=np.sum((y_pred != y)&(y_pred==0))
    return tp/(tp+fn)

def f_score(y_pred,y, beta_square=1):
    p=precision(y_pred,y)
    r=recall(y_pred,y)
    return (beta_square+1)*p*r/(beta_square*p+r)

def fpr(y_pred,y):
    fp=np.sum((y_pred != y)&(y_pred==1))
    tn=np.sum((y_pred == y)&(y_pred==0))
    return fp/(fp+tn)

def build_roc(model, X_test, y_test, n_tresholds=1000): #Построение ROC-кривой
    recalls=[]
    fprs=[]
    for i in range(1,n_tresholds):
        y_pred=model.predict_classes(X_test, i/n_tresholds)
        recalls.append(recall(y_pred,y_test))
        fprs.append(fpr(y_pred,y_test))
    plt.plot(fprs, recalls, color='b')
    plt.plot([i/n_tresholds for i in range(1,n_tresholds)], [i/n_tresholds for i in range(1,n_tresholds)], '--', color='g')
    plt.title("ROC-кривая")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.show()

def mae(y_pred, y):
  err = np.mean(np.abs(y_pred - y))
  return err

def dmae(y_pred, y):
  n = y.shape[0]
  return (1/n)*np.sign(y_pred - y)

def eWt(y_pred,y, t=0.02):
    return np.sum(np.abs(y_pred-y)<t)/len(y_pred)

def R_square(y_pred, y):
  y_mean = np.mean(y)
  return 1 - np.sum(np.square(y - y_pred))/np.sum(np.square(y - y_mean))

def mse(y_pred, y):
  err = np.mean(np.square(y_pred - y))
  return err

def dmse(y_pred, y):
  n = y.shape[0]
  return (2/n)*(y_pred - y)

def bce(y_pred, y):#Функция потерь - бинарная кросс-энтропия
    err=-np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
    return err

def dbce(y_pred, y):#Производная бинарной кросс-энтропии
    n = y.shape[0]
    return (1/n)*(y_pred-y)/(y_pred-np.power(y_pred,2))

def e(y_pred,y):
    return 1/2*np.sum(np.square(y_pred - y))

def de(y_pred,y):
    return y_pred-y

def rmse(y_pred, y):
  err = np.sqrt(np.mean(np.square(y_pred - y)))
  return err

def cross_entropy_loss(y_pred,y, n_classes=2):
    y_real=np.zeros([len(y),n_classes])
    for i in range(len(y)):
        y_real[i,y[i]]=1
    err=-np.sum(y_real*np.log(y_pred))/len(y)
    return err

def KLD(mu, sigma):
    return 0.5*np.sum(np.exp(sigma)+mu**2-1-sigma)

def dKLD(mu,sigma):
    return [mu,-0.5+0.5*np.exp(sigma)]
