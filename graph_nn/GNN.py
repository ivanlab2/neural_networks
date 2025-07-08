import numpy as np
from utils.metrics import mae, dmae, mse, R_square, eWt
from utils.activations import leakyrelu, dleakyrelu
rng = np.random.default_rng(51)


def get_batches(data, batch_size):
    n = len(data)
    get_X = lambda z: z[0]
    get_A = lambda z: z[1]
    get_shapes = lambda z: z[2]
    get_y = lambda z: z[3]
    for i in range(0, n, batch_size):
        batch = data[i:i+batch_size]
        yield np.array([get_X(b) for b in batch]), np.array([get_A(b) for b in batch]), np.array([get_shapes(b) for b in batch]), np.array([get_y(b) for b in batch])

def make_pad_mask(H,shapes):#Маска для зануления градиентов пэддингов строк
    h=np.zeros_like(H)
    for i in range(h.shape[0]):
        h[i,:shapes[i],:]=1
    return h

class GNN:
    def __init__(self, architecture):
        self.depth = len(architecture)-1
        # Функции активации и потерь
        self.activation_fn = leakyrelu 
        self.activation_dfn = dleakyrelu 
        self.loss_fn = mae 
        self.loss_dfn = dmae
        # Иницализация весов
        self.W, self.b = self._init_weights(architecture)
        # Выходы слоёв
        self.X = [None] * (self.depth+1)
        self.Z = [None] * (self.depth)
        self.H = [None] * (self.depth+1)
        # Градиенты
        self.dW=[np.zeros_like(w) for w in self.W]
        self.db=np.zeros_like(self.b)
        # Параметры Adam
        self.t = 1
        self.mW = [np.zeros_like(w) for w in self.W]
        self.mb = np.zeros_like(self.b)
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = np.zeros_like(self.b)

    def _init_weights(self, arch): #Инициализация весов и смещения
        net_in = arch[0]
        net_out = arch[-1]
        limit = np.sqrt(6. / (net_in + net_out))
        weights=[rng.uniform(-limit, limit + 1e-5, size=(arch[i], arch[i+1])) for i in range(self.depth)]
        weights.append(rng.uniform(-limit, limit + 1e-5, size=(arch[-1], 1)))
        bias=rng.random((1))*2-1
        return weights, bias    
        
    def _update_params_adam(self, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-7):#Обновление весов методом adam
        for i in range(self.depth+1):
            self.mW[i] = beta_1*self.mW[i] + (1-beta_1)*self.dW[i]
            self.vW[i] = beta_2*self.vW[i] + (1-beta_2)*(self.dW[i]**2)
            mW_corr = self.mW[i] / (1-beta_1**self.t)
            vW_corr = self.vW[i] / (1-beta_2**self.t)
            self.W[i] -= lr*mW_corr / (np.sqrt(vW_corr)+eps)
        self.mb = beta_1*self.mb + (1-beta_1)*self.db
        self.vb = beta_2*self.vb + (1-beta_2)*(self.db**2)
        mb_corr = self.mb / (1-beta_1**self.t)
        vb_corr = self.vb / (1-beta_2**self.t)
        self.b -= lr*mb_corr / (np.sqrt(vb_corr)+eps)
        # update time
        self.t += 1
        
    def _update_params_sgd(self, lr=1e-2): #Обновление весов методом SGD
        for i in range(self.depth+1):
            self.W[i] -= lr*self.dW[i]
        self.b -= lr*self.db
          
    def _feedforward(self, X, A, shapes):
        self.H[0]=X #Входная матрица признаков батча
        #Формирование стандартизованной матрицы смежности
        S=np.power(np.sum(A,axis=2),-0.5)
        S[S == np.inf] = 0
        D=np.apply_along_axis(np.diagflat,1,S)
        self.A_norm=np.matmul(np.matmul(D,A),D)
        #Проход через графовые свёрточные слои
        for i in range(self.depth):
            self.X[i]=np.matmul(self.A_norm,self.H[i])#Умножение стандартизованной матрицы смежности на вход в слой
            self.Z[i]=np.matmul(self.X[i],self.W[i])#Умножение на матрицу весов
            self.H[i+1]=self.activation_fn(self.Z[i])#Проход через слой активации
        self.avs=np.sum(self.H[-1],axis=1)/shapes.reshape(-1,1)#Global average pooling
        self.res=np.matmul(self.avs,self.W[-1])+self.b#Получение итогового результата с помощью полносвязного слоя

    def _compute_loss(self, X,A, shapes, y):
        y_pred = self.predict(X,A, shapes).reshape(y.shape)
        return self.loss_fn(y_pred, y), y_pred


    def _backprop(self, y, shapes):
        delta=self.loss_dfn(self.res,y)
        #Обратный проход через линейный слой
        self.dW[-1]=np.matmul(self.avs.T,delta)
        self.db=np.sum(delta)
        delta=np.matmul(delta,self.W[-1].T)
        #Обратный проход через global average pooling
        delta=np.repeat(delta/shapes.reshape(-1,1), self.H[-1].shape[1],axis=0).reshape(self.H[-1].shape)*make_pad_mask(self.H[-1],shapes)
        #Обратный проход через графовые сверточные слои
        for i in range(self.depth-1,-1,-1):
            delta=delta*self.activation_dfn(self.Z[i])#Обратный проход через слой активации
            self.dW[i]=np.sum(np.matmul(np.matmul(self.X[i].transpose(0,2,1),self.A_norm.transpose(0,2,1)),delta),axis=0)#Обновление весов
            delta=np.matmul(np.matmul(self.A_norm.transpose(0,2,1),delta),self.W[i].T)#Передача дельты на следующий слой

    def predict(self, X, A, shapes):
        H=X
        S=np.power(np.sum(A,axis=2),-0.5)
        S[S == np.inf] = 0
        D=np.apply_along_axis(np.diagflat,1,S)
        A_norm=np.matmul(np.matmul(D,A),D)
        for i in range(self.depth):
            X=np.matmul(A_norm,H)
            H=self.activation_fn(np.matmul(X,self.W[i]))
        avs=np.sum(H,axis=1)/shapes.reshape(-1,1)
        res=np.matmul(avs,self.W[-1])+self.b
        return res

    def train(self, X, A, shapes, y, X_test,A_test, shapes_test, y_test, epochs=1, exp_rate=1, batch_size=32, lr=0.001):
        epoch_losses = np.array([])
        dataset = list(zip(X, A, shapes, y))
        for i in range(epochs):
            rng.shuffle(dataset)
            for (X_batch, A_batch, shapes_batch, y_batch) in get_batches(dataset, batch_size):
                self._feedforward(X_batch, A_batch, shapes_batch)
                self._backprop(y_batch, shapes_batch)
                self._update_params_adam(lr)
            loss, y_pred=self._compute_loss(X_test,A_test, shapes_test, y_test)
            print('----------------------------------------------------------------------')
            print(f"Epoch: {i+1}")
            print('mse:',mse(y_pred,y_test))
            print('mae:',mae(y_pred,y_test))
            print('R^2:',R_square(y_pred,y_test))
            print('eWt:',eWt(y_pred,y_test))
            epoch_losses = np.append(epoch_losses, loss)
            lr=lr*exp_rate#Уменьшение на определенный коэффициент learning rate каждую эпоху
        return epoch_losses