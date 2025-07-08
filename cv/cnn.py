import numpy as np
rng = np.random.default_rng(51)
from utils.metrics import cross_entropy_loss
from utils.activations import relu, relu2deriv, d_softmax_cross_entropy
from utils.functions import im2col, col2im

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1)[:,None]

def get_batches(data, batch_size):
    n = len(data)
    get_X = lambda z: z[0]
    get_y = lambda z: z[1]
    for i in range(0, n, batch_size):
        batch = data[i:i+batch_size]
        yield np.array([get_X(b) for b in batch]), np.array([get_y(b) for b in batch])


class conv:
    def __init__(self, architecture):
        self.depth = len(architecture)-2
        #Функции активации и потерь
        self.activation_fn = relu 
        self.activation_dfn = relu2deriv 
        self.activation_end = softmax #Функция активации в последнем слое
        self.activation_dend = d_softmax_cross_entropy #Производная активации на последнем слое
        self.loss_fn = cross_entropy_loss 
       
        # Параметры модели
        self.W = self._init_weights(architecture)
        self.b = self._init_biases(architecture)
        self.res=[None]*(self.depth)#Результаты функции im2col
        self.z = [None] * (self.depth)
        self.m=[None] * (self.depth+1)#Входные значения слоёв
        self.a = [None] * (self.depth)
        self.masks=[None]*(self.depth)#Маски макспулинга
        # backprop gradients
        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b]
        # Adam optimizer params
        self.t = 1
        self.mW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(w) for w in self.b]

  
  # glorot uniform init
    def _init_weights(self, arch):
        W=[]
        s=np.sqrt(arch[0])
        net_in = arch[0]
        net_out = arch[-1]
        limit = np.sqrt(6. / (net_in + net_out))
        for i in range(1,self.depth+1):
            if i!=self.depth: #Инициализация весов для полносвязного слоя
                W.append(np.array(rng.uniform(-limit, limit + 1e-5, size=(4,arch[i],arch[i]))))
                s=int((s-arch[i]+1)//2)
            else:#Инициализация весов для свёрточных слоёв
                W.append(np.array(rng.uniform(-limit, limit + 1e-5, size=(arch[i],4*np.square(s)))))
        return W
        

    def _init_biases(self, arch):
        b=[]
        for i in range(1,self.depth+1):
            if i!=self.depth: 
                b.append(rng.random((4,1,1)))#Инициализация смещений для полносвязного слоя
            else:
                b.append(rng.random((arch[i],1)))#Инициализация смещений для свёрточных слоёв
        return b
        



    def feedforward(self, X):#Прямой проход
        self.m[0]=np.repeat(X,4,axis=1).reshape(X.shape[0],4,X.shape[1],X.shape[2],order='F')
        for i in range(self.depth-1):
            self.res[i]=im2col(self.m[i],(self.W[i].shape[1],self.W[i].shape[1]))#Развертывание картинки в столбец
            self.a[i]=(np.matmul(self.W[i].reshape(self.W[i].shape[0],1,-1), self.res[i])+self.b[i]).reshape(self.m[i].shape[0],self.m[i].shape[1],-1)#Получение результатов свёртки (обычное матричное произведение)
            self.z[i]=self.activation_fn(self.a[i])#Проход через слой активации
            self.m[i+1]=self.z[i].reshape(self.z[i].shape[0],self.z[i].shape[1],(self.m[i].shape[2]-self.W[i].shape[1]+1)//2, 2, (self.m[i].shape[2]-self.W[i].shape[2]+1)//2, 2).max((3, 5))#Максипулинг
            self.masks[i]=np.isclose(self.z[i].reshape(self.z[i].shape[0],self.z[i].shape[1],self.m[i].shape[2]-self.W[i].shape[1]+1,self.m[i].shape[2]-self.W[i].shape[1]+1),np.repeat(np.repeat(self.m[i+1],2,axis=3),2,axis=2))#Маска макспулинга
        self.m[2]=self.m[2].reshape(self.m[2].shape[0],-1)
        self.z[2] = np.matmul(self.m[2], self.W[2].T)+self.b[-1].T#Проход через полносвязный слой
        self.m[-1]=self.activation_end(self.z[2])#Получение итогового результата
       

    def _compute_loss(self, X, y):
        y_pred = self.predict(X)
        return self.loss_fn(y_pred, y, n_classes=10)


    def backprop(self, y, batch_size=32):#Обратный проход
        delta=self.activation_dend(self.m[-1],y,n_classes=10)#Получение дельты софтмакса и потерь
        #Проход через полносвязный слой
        self.dW[-1] = np.matmul(delta.T, self.m[-2])
        self.db[-1] = np.sum(delta, axis=0, keepdims=True).T
        delta=np.matmul(delta, self.W[-1])
        #Превращение дельты в результат последнего макспулинга
        delta=np.reshape(delta, (delta.shape[0], 4,np.sqrt(delta.shape[1]//4).astype(np.int16),np.sqrt(delta.shape[1]//4).astype(np.int16)))
        for i in range(self.depth-2,-1,-1):
            delta=(np.repeat(np.repeat(delta,2,axis=3),2,axis=2)*self.masks[i]).reshape(delta.shape[0],delta.shape[1],-1)#Проход через слой пулинга
            delta=delta*self.activation_dfn(self.z[i])#Проход через слой активации
            self.db[i]=np.mean(np.sum(delta,axis=2),axis=0).reshape(4,1,1)#Получение дельт смещений
            self.dW[i]=np.mean(np.matmul(delta.reshape(delta.shape[0],delta.shape[1],1,-1),np.transpose(self.res[i], (0,1, 3, 2))).reshape(delta.shape[0],delta.shape[1],-1),axis=0).reshape(self.W[i].shape[0],self.W[i].shape[1],self.W[i].shape[2])#Обновление весов
            delta=col2im(np.matmul(self.W[i].reshape(self.W[i].shape[0],-1,1),delta.reshape(delta.shape[0],delta.shape[1],1,-1)),(self.m[i].shape[2],self.m[i].shape[2]),(self.W[i].shape[1],self.W[i].shape[1]))#Обновление дельты для макспулинга на следующем слое         


  
    def _update_params_sgd(self, lr=1e-2): 
        for i in range(self.depth):
            self.W[i] -= lr*self.dW[i]
            self.b[i] -= lr*self.db[i]
            
    def _update_params(self, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-7):
        for i in range(self.depth):
      # update first moments
            self.mW[i] = beta_1*self.mW[i] + (1-beta_1)*self.dW[i]
            self.mb[i] = beta_1*self.mb[i] + (1-beta_1)*self.db[i]
      # update second moments
            self.vW[i] = beta_2*self.vW[i] + (1-beta_2)*(self.dW[i]**2)
            self.vb[i] = beta_2*self.vb[i] + (1-beta_2)*(self.db[i]**2)
      # correction
            mW_corr = self.mW[i] / (1-beta_1**self.t)
            mb_corr = self.mb[i] / (1-beta_1**self.t)
            vW_corr = self.vW[i] / (1-beta_2**self.t)
            vb_corr = self.vb[i] / (1-beta_2**self.t)
      # update params
            self.W[i] -= lr*mW_corr / (np.sqrt(vW_corr)+eps)
            self.b[i] -= lr*mb_corr / (np.sqrt(vb_corr)+eps)
    # update time
            self.t += 1

    
    def train(self, X, y, epochs=1, batch_size=32, lr=0.0001):
        n = y.shape[0]
        epoch_losses = np.array([])
        dataset = list(zip(X, y))
        k=0
        for i in range(epochs):
            rng.shuffle(dataset)
            k+=1
            c=0
            for (X_batch, y_batch) in get_batches(dataset, batch_size):
                self.feedforward(X_batch)
                self.backprop(y_batch)
                self._update_params(lr=lr)#Лучше всего подошел adam c lr=0.0001
                c+=1
                if c%1000==0:
                    print(self._compute_loss(X_batch,y_batch))
            epoch_losses = np.append(epoch_losses, self._compute_loss(X[:1000], y[:1000]))
        return epoch_losses

    def predict(self, X):
        t=np.repeat(X,4,axis=1).reshape(X.shape[0],4,X.shape[1],X.shape[2],order='F')
        for i in range(self.depth-1):
            res=im2col(t,(self.W[i].shape[1],self.W[i].shape[1]))
            a=(np.matmul(self.W[i].reshape(self.W[i].shape[0],1,-1), res)+self.b[i]).reshape(t.shape[0],t.shape[1],-1)
            z=self.activation_fn(a)
            t=z.reshape(z.shape[0],z.shape[1],(t.shape[2]-self.W[i].shape[1]+1)//2, 2, (t.shape[2]-self.W[i].shape[2]+1)//2, 2).max((3, 5))
        t=t.reshape(t.shape[0],-1)
        z = np.matmul(t, self.W[2].T)+self.b[-1].T
        t=self.activation_end(z)
        return t
    
    def predict_classes(self, X): #Функция для расчёта строгого значения
        t=self.predict(X)
        max_t = t.reshape(t.shape[0], -1).argmax(1)
        maxpos = np.column_stack(np.unravel_index(max_t, t[0,: ].shape))
        p=maxpos.reshape(-1)
        return p
    
    def predict_binary(self,X, digit,p):#Функция для бинарного предсказания для конкретной цифры с конкретным порогом
        return self.predict(X)[:,digit]>p
        
        