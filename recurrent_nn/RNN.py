import numpy as np
from utils.activations import tanh, dtanh, sigm, dsigm
from utils.metrics import e, de
rng = np.random.default_rng(51)


class RMLP:
    def __init__(self, architecture):
        #Функции активации и потерь
        self.activation_fn = tanh 
        self.activation_dfn = dtanh 
        self.loss_fn = e 
        self.loss_dfn = de
        #Инициализация весов и смещений
        self.Wx,self.Wh,self.Wy  = self._init_weights(architecture)
        self.bh, self.by = self._init_biases(architecture)
        #Градиенты для бэкпропа
        self.dWx=np.zeros_like(self.Wx)
        self.dWy=np.zeros_like(self.Wy)
        self.dby=np.zeros_like(self.by)
        # Adam optimizer params
        self.t = 1
        self.mWx = np.zeros_like(self.Wx)
        self.mWh = np.zeros_like(self.Wh)
        self.mWy = np.zeros_like(self.Wy)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
        self.vWx = np.zeros_like(self.Wx)
        self.vWh = np.zeros_like(self.Wh)
        self.vWy = np.zeros_like(self.Wy)
        self.vbh = np.zeros_like(self.bh)
        self.vby = np.zeros_like(self.by)
  # glorot uniform init
    def _init_weights(self, arch):
        net_in = arch[0]
        net_out = arch[-1]
        limit = np.sqrt(6. / (net_in + net_out))
        Wx=rng.uniform(-limit, limit + 1e-5, size=(arch[0], arch[1]))
        Wh=rng.uniform(-limit, limit + 1e-5, size=(arch[1], arch[1]))
        Wy=rng.uniform(-limit, limit + 1e-5, size=(arch[1], arch[-1]))
        return Wx, Wh, Wy

    def _init_biases(self, arch):
        bh = rng.random((1,arch[1]))
        by = rng.random((1, arch[-1]))
        return bh,by

    def _feedforward(self, X):
        self.a=X
        self.h=np.zeros([X.shape[0],self.Wh.shape[0]])#Формирование матрицы со скрытыми состояниями
        self.outputs=np.zeros(self.a.shape[0])#Выходы модели
        for i in range(self.a.shape[0]):
            #Формирование новых скрытых состояний
            if i==0:
                self.h[i,:]=self.activation_fn(np.matmul(self.a[i,:],self.Wx)+self.bh)
            else:
                self.h[i,:]=self.activation_fn(np.matmul(self.a[i,:],self.Wx)+np.matmul(self.Wh,self.h[i-1,:])+self.bh)
            #Получение выхода
            self.outputs[i]=np.matmul(self.h[i,:],self.Wy)+self.by
            
    def _compute_loss(self, X, y):
        y_pred = self.predict_2(X).reshape(y.shape)
        return self.loss_fn(y_pred, y)

    def _backprop(self, y, batch_size=4):
        self.dWh=np.zeros_like(self.Wh)
        self.dbh=np.zeros_like(self.bh)
        deltas=[np.zeros_like(self.h),np.zeros_like(self.outputs)]
        deltas[1]=self.loss_dfn(self.outputs,y)#Дельта полносвязного слоя
        #Обновление весов и смещения полносвязного слоя
        self.dWy=np.matmul(self.h.T,deltas[1].reshape(-1,1))
        self.dby = np.sum(deltas[1])
        for i in range(batch_size-1,0,-1):
        #Получение дельт рекурентного слоя через BPTT
            if i==batch_size-1:
                deltas[0][i,:]=self.activation_dfn(self.h[i,:])*(deltas[1][i]*self.Wy.T)[0]
            else:
                deltas[0][i,:]=self.activation_dfn(self.h[i,:])*((deltas[1][i]*self.Wy.T)[0]+np.matmul(deltas[0][i+1,:],self.Wh.T))
        self.dWx=np.matmul(self.a.T,deltas[0])
        for i in range(1,batch_size):
        #Обновление рекурентных слоёв
            self.dWh += np.outer(deltas[0][i], self.h[i-1])
            self.dbh+=deltas[0][i-1,:]
  
    def _update_params_sgd(self, lr=1e-2): 
        self.Wx -= lr*self.dWx
        self.Wh -= lr*self.dWh
        self.Wy -= lr*self.dWy
        self.bh -= lr*self.dbh
        self.by -= lr*self.dby
        
    def _update_params(self, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-7):
      # update first moments
        self.mWx = beta_1*self.mWx + (1-beta_1)*self.dWx
        self.mWh = beta_1*self.mWh + (1-beta_1)*self.dWh
        self.mWy = beta_1*self.mWy + (1-beta_1)*self.dWy
        self.mbh = beta_1*self.mbh + (1-beta_1)*self.dbh
        self.mby = beta_1*self.mby + (1-beta_1)*self.dby
      # update second moments
        self.vWx = beta_2*self.vWx + (1-beta_2)*(self.dWx**2)
        self.vWh = beta_2*self.vWh + (1-beta_2)*(self.dWh**2)
        self.vWy = beta_2*self.vWy + (1-beta_2)*(self.dWy**2)
        self.vbh = beta_2*self.vbh + (1-beta_2)*(self.dbh**2)
        self.vby = beta_2*self.vby + (1-beta_2)*(self.dby**2)
      # correction
        mWx_corr = self.mWx / (1-beta_1**self.t)
        mWh_corr = self.mWh / (1-beta_1**self.t)
        mWy_corr = self.mWy / (1-beta_1**self.t)
        mbh_corr = self.mbh / (1-beta_1**self.t)
        mby_corr = self.mby / (1-beta_1**self.t)
        vWx_corr = self.vWx / (1-beta_2**self.t)
        vWh_corr = self.vWh / (1-beta_2**self.t)
        vWy_corr = self.vWy / (1-beta_2**self.t)
        vbh_corr = self.vbh / (1-beta_2**self.t)
        vby_corr = self.vby / (1-beta_2**self.t)
      # update params
        self.Wx -= lr*mWx_corr / (np.sqrt(vWx_corr)+eps)
        self.Wh -= lr*mWh_corr / (np.sqrt(vWh_corr)+eps)
        self.Wy -= lr*mWy_corr / (np.sqrt(vWy_corr)+eps)
        self.bh -= lr*mbh_corr / (np.sqrt(vbh_corr)+eps)
        self.by -= lr*mby_corr / (np.sqrt(vby_corr)+eps)
    # update time
        self.t += 1

    def train(self, X, Y, epochs=1, batch_size=4):
        #Разбиваем тренировочную выборку по батчам размера 4 по умолчанию
        x=X.reshape(int(X.shape[0]/batch_size),batch_size,X.shape[1])
        y=Y.reshape(-1,batch_size)
        epoch_losses = np.array([])
        dataset = list(zip(x, y))
        for i in range(epochs):
            rng.shuffle(dataset)
            for (x_b, y_b) in dataset:
                self._feedforward(x_b)
                self._backprop(y_b)
                self._update_params(lr=0.0001)
            epoch_losses = np.append(epoch_losses, self._compute_loss(X, Y))
        return epoch_losses

    def predict(self, x):#Предсказание на всём ряду
        predicts=np.array([])
        h_prev=np.zeros(self.Wh.shape[0])
        for X in x:
            h=self.activation_fn(np.matmul(X,self.Wx)+np.matmul(self.Wh,h_prev)+self.bh)
            h_prev=h.reshape(-1)
            predicts=np.append(predicts,(np.matmul(h,self.Wy)+self.by)[0])
        return predicts
    
    def predict_2(self, X_b, b_size=4):#Предсказание на батчах (нужно при обучении)
        x=X_b.reshape(int(X_b.shape[0]/b_size),b_size,X_b.shape[1])
        predicts=np.array([])
        for X in x:
            a=X
            h_prev=np.zeros(self.Wh.shape[0])
            for i in range(a.shape[0]):
                if i==0:
                    h=self.activation_fn(np.matmul(a[i,:],self.Wx)+self.bh)
                else:
                    h=self.activation_fn(np.matmul(a[i,:],self.Wx)+np.matmul(self.Wh,h_prev)+self.bh)
                h_prev=h.reshape(-1)
                predicts=np.append(predicts,(np.matmul(h,self.Wy)+self.by)[0])
        return predicts
    
    
class LSTM:
    def __init__(self, architecture):
        # functions
        self.loss_fn = e 
        self.loss_dfn = de 
        # Инициализация весов и смещений
        self.Wfh, self.Wfx,self.Wih,self.Wix, self.Wch,self.Wcx,self.Woh,self.Wox,self.Wy  = self._init_weights(architecture)
        self.bf,self.bi, self.bc,self.bo,self.by = self._init_biases(architecture)
        # Adam optimizer params
        self.t = 1
        self.mWfx = np.zeros_like(self.Wfx)
        self.mWix = np.zeros_like(self.Wix)
        self.mWcx = np.zeros_like(self.Wcx)
        self.mWox = np.zeros_like(self.Wox)
        self.mWfh = np.zeros_like(self.Wfh)
        self.mWih = np.zeros_like(self.Wih)
        self.mWch = np.zeros_like(self.Wch)
        self.mWoh = np.zeros_like(self.Woh)
        self.mWy = np.zeros_like(self.Wy)
        self.mbf = np.zeros_like(self.bf)
        self.mbi = np.zeros_like(self.bi)
        self.mbc = np.zeros_like(self.bc)
        self.mbo = np.zeros_like(self.bo)
        self.mby = np.zeros_like(self.by)
        self.vWfx = np.zeros_like(self.Wfx)
        self.vWix = np.zeros_like(self.Wix)
        self.vWcx = np.zeros_like(self.Wcx)
        self.vWox = np.zeros_like(self.Wox)
        self.vWfh = np.zeros_like(self.Wfh)
        self.vWih = np.zeros_like(self.Wih)
        self.vWch = np.zeros_like(self.Wch)
        self.vWoh = np.zeros_like(self.Woh)
        self.vWy = np.zeros_like(self.Wy)
        self.vbf = np.zeros_like(self.bf)
        self.vbi = np.zeros_like(self.bi)
        self.vbc = np.zeros_like(self.bc)
        self.vbo = np.zeros_like(self.bc)
        self.vby = np.zeros_like(self.by)
        
  # glorot uniform init
    def _init_weights(self, arch):
        net_in = arch[0]
        net_out = arch[-1]
        limit = np.sqrt(6. / (net_in + net_out))
        Wfh=rng.uniform(-limit, limit + 1e-5, size=(arch[1], arch[1]))
        Wfx=rng.uniform(-limit, limit + 1e-5, size=(arch[0], arch[1]))
        Wih=rng.uniform(-limit, limit + 1e-5, size=(arch[1], arch[1]))
        Wix=rng.uniform(-limit, limit + 1e-5, size=(arch[0], arch[1]))
        Wch=rng.uniform(-limit, limit + 1e-5, size=(arch[1], arch[1]))
        Wcx=rng.uniform(-limit, limit + 1e-5, size=(arch[0], arch[1]))
        Woh=rng.uniform(-limit, limit + 1e-5, size=(arch[1], arch[1]))
        Wox=rng.uniform(-limit, limit + 1e-5, size=(arch[0], arch[1]))
        Wy=rng.uniform(-limit, limit + 1e-5, size=(arch[1], arch[-1]))
        return Wfh, Wfx,Wih,Wix, Wch,Wcx,Woh,Wox,Wy


    def _init_biases(self, arch):
        bf = rng.random((1,arch[1]))
        bi = rng.random((1,arch[1]))
        bc = rng.random((1,arch[1]))
        bo= rng.random((1,arch[1]))
        by=rng.random((1, arch[-1]))
        return bf,bi, bc,bo,by


    def _feedforward(self, X):
        self.a=X
        self.h=np.zeros([self.a.shape[0],self.Wfh.shape[0]])
        self.f=np.zeros([self.a.shape[0],self.Wfh.shape[0]])
        self.inp=np.zeros([self.a.shape[0],self.Wfh.shape[0]])
        self.C_cand=np.zeros([self.a.shape[0],self.Wfh.shape[0]])
        self.C=np.zeros([self.a.shape[0],self.Wfh.shape[0]])
        self.o=np.zeros([self.a.shape[0],self.Wfh.shape[0]])
        for i in range(self.a.shape[0]):
            #Получение прошлых ячеек памяти и скрытых состояний
            if i==0:
                h_prev=np.zeros(self.Wfh.shape[0])
                C_prev=np.zeros(self.Wfh.shape[0])
            else:
                h_prev=self.h[i-1,:]
                C_prev=self.C[i-1,:]
            self.f[i,:]=sigm(np.matmul(self.a[i,:],self.Wfx)+np.matmul(self.Wfh,h_prev)+self.bf)#Фильтр забывания
            self.inp[i,:]=sigm(np.matmul(self.a[i,:],self.Wix)+np.matmul(self.Wih,h_prev)+self.bi)#Слой входного фильтра
            self.C_cand[i,:]=tanh(np.matmul(self.a[i,:],self.Wcx)+np.matmul(self.Wch,h_prev)+self.bc)#Формирование вектора-кандидата состояния ячейки
            self.C[i,:]=(C_prev*self.f[i,:]+self.C_cand[i,:]*self.inp[i,:])#Обновление состояния ячейки
            self.o[i,:]=sigm(np.matmul(self.a[i,:],self.Wox)+np.matmul(self.Woh,h_prev)+self.bo)#Слой выходного фильтра
            self.h[i,:]=(self.o[i,:]*tanh(self.C[i,:]))#Формирование нового скрытого состояния   
        self.outputs=(np.matmul(self.h,self.Wy)+self.by).reshape(-1)#Проход через полносвязный слой  
        
    def _compute_loss(self, X, y):
        y_pred = self.predict_2(X)
        return self.loss_fn(y_pred, y)


    def _backprop(self, y, batch_size=4):
        self.dWfh=np.zeros_like(self.Wfh)
        self.dWfx=np.zeros_like(self.Wfx)
        self.dWih=np.zeros_like(self.Wih)
        self.dWix=np.zeros_like(self.Wix)
        self.dWch=np.zeros_like(self.Wch)
        self.dWcx=np.zeros_like(self.Wcx)
        self.dWoh=np.zeros_like(self.Woh)
        self.dWox=np.zeros_like(self.Wox)
        self.dWy=np.zeros_like(self.Wy)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo= np.zeros_like(self.bo)
        self.dby=np.zeros_like(self.by)
        #Обратный проход черех полносвязный слой
        doutput=self.loss_dfn(self.outputs,y)
        self.dWy=np.matmul(self.h.T,doutput).reshape(-1,1)
        self.dby = np.sum(doutput)
        #Проход через реккурентный слой (метод BPTT)
        delta_o=np.zeros_like(self.h)#Дельта выходного слоя
        delta_i=np.zeros_like(self.h)#Дельта входного слоя
        delta_c=np.zeros_like(self.h)#Дельта слоя кандидатов
        delta_f=np.zeros_like(self.h)#Дельта слоя забывания
        for i in range(batch_size-1,-1,-1):
        #Получение дельт следующей временной отметки
            if i==batch_size-1:
                delta_o_next=np.zeros(self.Wfh.shape[0])
                delta_c_next=np.zeros(self.Wfh.shape[0])
                delta_i_next=np.zeros(self.Wfh.shape[0])
                delta_f_next=np.zeros(self.Wfh.shape[0])
            else:
                delta_o_next=delta_o[i+1]
                delta_c_next=delta_c[i+1]
                delta_i_next=delta_i[i+1]
                delta_f_next=delta_f[i+1]
            delta_o[i,:]=tanh(self.C[i,:])*dsigm(self.o[i,:])*((doutput[i]*self.Wy.T)[0]+np.matmul(delta_o_next,self.Woh.T))#Градиент для выходного слоя
            delta_c[i,:]=dtanh(self.C_cand[i,:])*(dtanh(self.C[i,:])*self.o[i,:]*delta_o[i,:]+np.matmul(delta_c_next,self.Wch.T))#Градиент для слоя кандидатов
            delta_i[i,:]=dsigm(self.inp[i,:])*(dtanh(self.C[i,:])*self.o[i,:]*delta_o[i,:]+np.matmul(delta_i_next,self.Wih.T))#Градиент для входного слоя
            delta_f[i,:]=dsigm(self.f[i,:])*(dtanh(self.C[i,:])*self.o[i,:]*delta_o[i,:]+np.matmul(delta_f_next,self.Wfh.T))#Градиент для слоя забывания
            #Получение дельт на весах
        for i in range(1,batch_size):
            self.dWfh+=np.matmul(delta_f[i,:].reshape(-1,1),self.h[i-1,:].reshape(1,-1))
            self.dWfx+np.matmul(self.a[i-1,:].reshape(-1,1),delta_f[i,:].reshape(1,-1))
            self.dWih+=np.matmul(delta_i[i,:].reshape(-1,1),self.h[i-1,:].reshape(1,-1))
            self.dWix+=np.matmul(self.a[i-1,:].reshape(-1,1),delta_i[i,:].reshape(1,-1))
            self.dWch+=np.matmul(delta_c[i,:].reshape(-1,1),self.h[i-1,:].reshape(1,-1))
            self.dWcx+=np.matmul(self.a[i-1,:].reshape(-1,1),delta_c[i,:].reshape(1,-1))
            self.dWoh+np.matmul(delta_o[i,:].reshape(-1,1),self.h[i-1,:].reshape(1,-1))
            self.dWox+=np.matmul(self.a[i-1,:].reshape(-1,1),delta_o[i,:].reshape(1,-1))
            self.dbf+= delta_f[i,:]
            self.dbi+= delta_i[i,:]
            self.dbc+= delta_c[i,:]
            self.dbo+= delta_o[i,:]

  
    def _update_params_sgd(self, lr=1e-2): 
        self.Wfh-=lr*self.dWfh
        self.Wfx-=lr*self.dWfx
        self.Wih-=lr*self.dWih
        self.Wix-=lr*self.dWix
        self.Wch-=lr*self.dWch
        self.Wcx-=lr*self.dWcx
        self.Woh-=lr*self.dWoh
        self.Wox-=lr*self.dWox
        self.Wy-=lr*self.dWy
        self.bf -= lr*self.dbf 
        self.bi -= lr*self.dbi
        self.bc -= lr*self.dbc
        self.bo-= lr*self.dbo
        self.by-=lr*self.dby
        
    def _update_params(self, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-7):
      # update first moments
        self.mWfh = beta_1*self.mWfh + (1-beta_1)*self.dWfh
        self.mWfx = beta_1*self.mWfx + (1-beta_1)*self.dWfx
        self.mWih = beta_1*self.mWih + (1-beta_1)*self.dWih
        self.mWix = beta_1*self.mWix + (1-beta_1)*self.dWix
        self.mWch = beta_1*self.mWch + (1-beta_1)*self.dWch
        self.mWcx = beta_1*self.mWcx + (1-beta_1)*self.dWcx
        self.mWoh = beta_1*self.mWoh + (1-beta_1)*self.dWoh
        self.mWox = beta_1*self.mWox + (1-beta_1)*self.dWox
        self.mWy = beta_1*self.mWy + (1-beta_1)*self.dWy
        self.mbf = beta_1*self.mbf + (1-beta_1)*self.dbf
        self.mbi = beta_1*self.mbi + (1-beta_1)*self.dbi
        self.mbc = beta_1*self.mbc + (1-beta_1)*self.dbc
        self.mbo = beta_1*self.mbo + (1-beta_1)*self.dbo
        self.mby = beta_1*self.mby + (1-beta_1)*self.dby
      # update second moments
        self.vWfh = beta_2*self.vWfh + (1-beta_2)*(self.dWfh**2)
        self.vWfx = beta_2*self.vWfx + (1-beta_2)*(self.dWfx**2)
        self.vWih = beta_2*self.vWih + (1-beta_2)*(self.dWih**2)
        self.vWch = beta_2*self.vWch + (1-beta_2)*(self.dWch**2)
        self.vWcx = beta_2*self.vWcx + (1-beta_2)*(self.dWcx**2)
        self.vWoh = beta_2*self.vWoh + (1-beta_2)*(self.dWoh**2)
        self.vWox = beta_2*self.vWox + (1-beta_2)*(self.dWox**2)
        self.vWih = beta_2*self.vWih + (1-beta_2)*(self.dWih**2)
        self.vWy = beta_2*self.vWy + (1-beta_2)*(self.dWy**2)
        self.vbf = beta_2*self.vbf + (1-beta_2)*(self.dbf**2)
        self.vbi = beta_2*self.vbi + (1-beta_2)*(self.dbi**2)
        self.vbc = beta_2*self.vbc + (1-beta_2)*(self.dbc**2)
        self.vbo = beta_2*self.vbo + (1-beta_2)*(self.dbo**2)
        self.vby = beta_2*self.vby + (1-beta_2)*(self.dby**2)
      # correction
        mWfh_corr = self.mWfh / (1-beta_1**self.t)
        mWfx_corr = self.mWfx / (1-beta_1**self.t)
        mWih_corr = self.mWih / (1-beta_1**self.t)
        mWch_corr = self.mWch / (1-beta_1**self.t)
        mWcx_corr = self.mWcx / (1-beta_1**self.t)
        mWoh_corr = self.mWoh / (1-beta_1**self.t)
        mWox_corr = self.mWox / (1-beta_1**self.t)
        mWih_corr = self.mWih / (1-beta_1**self.t)
        mWy_corr = self.mWy / (1-beta_1**self.t)
        mbf_corr = self.mbf / (1-beta_1**self.t)
        mbi_corr = self.mbi / (1-beta_1**self.t)
        mbc_corr = self.mbc / (1-beta_1**self.t)
        mbo_corr = self.mbo / (1-beta_1**self.t)
        mby_corr = self.mby / (1-beta_1**self.t)
        vWfh_corr = self.vWfh / (1-beta_2**self.t)
        vWfx_corr = self.vWfx / (1-beta_2**self.t)
        vWih_corr = self.vWih / (1-beta_2**self.t)
        vWch_corr = self.vWch / (1-beta_2**self.t)
        vWcx_corr = self.vWcx / (1-beta_2**self.t)
        vWoh_corr = self.vWoh / (1-beta_2**self.t)
        vWih_corr = self.vWih / (1-beta_2**self.t)
        vWox_corr = self.vWox / (1-beta_2**self.t)
        vWy_corr = self.vWy / (1-beta_2**self.t)
        vbf_corr = self.vbf / (1-beta_2**self.t)
        vbi_corr = self.vbi / (1-beta_2**self.t)
        vbc_corr = self.vbc / (1-beta_2**self.t)
        vbo_corr = self.vbo / (1-beta_2**self.t)
        vby_corr = self.vby / (1-beta_2**self.t)
      # update params
        self.Wfh -= lr*mWfh_corr / (np.sqrt(vWfh_corr)+eps)
        self.Wfx -= lr*mWfx_corr / (np.sqrt(vWfx_corr)+eps)
        self.Wih -= lr*mWih_corr / (np.sqrt(vWih_corr)+eps)
        self.Wch -= lr*mWch_corr / (np.sqrt(vWch_corr)+eps)
        self.Wcx -= lr*mWcx_corr / (np.sqrt(vWcx_corr)+eps)
        self.Woh -= lr*mWoh_corr / (np.sqrt(vWoh_corr)+eps)
        self.Wih -= lr*mWih_corr / (np.sqrt(vWih_corr)+eps)
        self.Wox -= lr*mWox_corr / (np.sqrt(vWox_corr)+eps)
        self.Wy -= lr*mWy_corr / (np.sqrt(vWy_corr)+eps)
        self.bf -= lr*mbf_corr / (np.sqrt(vbf_corr)+eps)
        self.bi -= lr*mbi_corr / (np.sqrt(vbi_corr)+eps)
        self.bc -= lr*mbc_corr / (np.sqrt(vbc_corr)+eps)
        self.bo -= lr*mbo_corr / (np.sqrt(vbo_corr)+eps)
        self.by -= lr*mby_corr / (np.sqrt(vby_corr)+eps)
    # update time
        self.t += 1


    def train(self, X, Y, epochs=1, batch_size=4):
        #Разбиение на мини-батчи
        x=X.reshape(int(X.shape[0]/batch_size),batch_size,X.shape[1])
        y=Y.reshape(-1,batch_size)
        epoch_losses = np.array([])
        dataset = list(zip(x, y))
        for i in range(epochs):
            rng.shuffle(dataset)
            for (x_b, y_b) in dataset:
                self._feedforward(x_b)
                self._backprop(y_b)
                self._update_params(lr=0.001)
            epoch_losses = np.append(epoch_losses, self._compute_loss(X, Y))
        return epoch_losses


    def predict(self, x):#Предсказание на всём временном ряду
        a=x
        h_prev=np.zeros(self.Wfh.shape[0])
        C_prev=np.zeros(self.Wfh.shape[0])
        predicts=np.array([])
        for i in range(a.shape[0]):
            f=sigm(np.matmul(a[i,:],self.Wfx)+np.matmul(self.Wfh,h_prev)+self.bf)
            inp=sigm(np.matmul(a[i,:],self.Wix)+np.matmul(self.Wih,h_prev)+self.bi)
            C_cand=tanh(np.matmul(a[i,:],self.Wcx)+np.matmul(self.Wch,h_prev)+self.bc)
            C=(C_prev*f+C_cand*inp)
            o=sigm(np.matmul(a[i,:],self.Wox)+np.matmul(self.Woh,h_prev)+self.bo)
            h=(o*tanh(C))
            C_prev=C.reshape(-1)
            h_prev=h.reshape(-1)
            predicts=np.append(predicts, (np.matmul(h,self.Wy)+self.by).reshape(-1))
        return predicts   
    
    def predict_2(self, X_b, b_size=4):#Предсказание на мини-батчах (для обучения)
        X=X_b.reshape(int(X_b.shape[0]/b_size),b_size,X_b.shape[1])
        predicts=np.array([])
        for x in X:
            a=x
            h_prev=np.zeros(self.Wfh.shape[0])
            C_prev=np.zeros(self.Wfh.shape[0])
            for i in range(a.shape[0]):
                f=sigm(np.matmul(a[i,:],self.Wfx)+np.matmul(self.Wfh,h_prev)+self.bf)#Фильтр забывания
                inp=sigm(np.matmul(a[i,:],self.Wix)+np.matmul(self.Wih,h_prev)+self.bi)#Слой входного фильтра
                C_cand=tanh(np.matmul(a[i,:],self.Wcx)+np.matmul(self.Wch,h_prev)+self.bc)#Формирование вектора-кандидата состояния ячейкт
                C=(C_prev*f+C_cand*inp)#Обновление состояния ячейки
                o=sigm(np.matmul(a[i,:],self.Wox)+np.matmul(self.Woh,h_prev)+self.bo)#Слой выходного фильтра
                h=(o*tanh(C))#Формирование нового скрытого состояния   
                C_prev=C.reshape(-1)
                h_prev=h.reshape(-1) 
                pred=(np.matmul(h,self.Wy)+self.by).reshape(-1)  
                predicts=np.append(predicts,pred)
        return predicts
    
    
class GRU:
    def __init__(self, architecture):
        # functions
        self.loss_fn = e
        self.loss_dfn = de
        #Инициализация весов и смещений
        self.Wzh, self.Wzx,self.Wrh,self.Wrx, self.Whh,self.Whx,self.Wy  = self._init_weights(architecture)
        self.bz,self.br,self.bh,self.by = self._init_biases(architecture)

        # Adam optimizer params
        self.t = 1
        self.mWzx = np.zeros_like(self.Wzx)
        self.mWrx = np.zeros_like(self.Wrx)
        self.mWhx = np.zeros_like(self.Whx)
        self.mWzh = np.zeros_like(self.Wzh)
        self.mWrh = np.zeros_like(self.Wrh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWy = np.zeros_like(self.Wy)
        self.mbz = np.zeros_like(self.bz)
        self.mbr = np.zeros_like(self.br)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
        self.vWzx = np.zeros_like(self.Wzx)
        self.vWrx = np.zeros_like(self.Wrx)
        self.vWhx = np.zeros_like(self.Whx)
        self.vWzh = np.zeros_like(self.Wzh)
        self.vWrh = np.zeros_like(self.Wrh)
        self.vWhh = np.zeros_like(self.Whh)
        self.vWy = np.zeros_like(self.Wy)
        self.vbz = np.zeros_like(self.bz)
        self.vbr = np.zeros_like(self.br)
        self.vbh = np.zeros_like(self.bh)
        self.vby = np.zeros_like(self.by)

  
  # glorot uniform init
    def _init_weights(self, arch):
        net_in = arch[0]
        net_out = arch[-1]
        limit = np.sqrt(6. / (net_in + net_out))
        Wzh=rng.uniform(-limit, limit + 1e-5, size=(arch[1], arch[1]))
        Wzx=rng.uniform(-limit, limit + 1e-5, size=(arch[0], arch[1]))
        Wrh=rng.uniform(-limit, limit + 1e-5, size=(arch[1], arch[1]))
        Wrx=rng.uniform(-limit, limit + 1e-5, size=(arch[0], arch[1]))
        Whh=rng.uniform(-limit, limit + 1e-5, size=(arch[1], arch[1]))
        Whx=rng.uniform(-limit, limit + 1e-5, size=(arch[0], arch[1]))
        Wy=rng.uniform(-limit, limit + 1e-5, size=(arch[1], arch[-1]))
        return Wzh, Wzx,Wrh,Wrx, Whh,Whx,Wy


    def _init_biases(self, arch):
        bz = rng.random((1,arch[1]))
        br = rng.random((1,arch[1]))
        bh = rng.random((1,arch[1]))
        by=rng.random((1, arch[-1]))
        return bz,br,bh,by


    def _feedforward(self, X):
        self.a=X
        self.h=np.zeros([self.a.shape[0],self.Whh.shape[0]])
        self.z=np.zeros([self.a.shape[0],self.Whh.shape[0]])
        self.r=np.zeros([self.a.shape[0],self.Whh.shape[0]])
        self.h_cand=np.zeros([self.a.shape[0],self.Whh.shape[0]])
        for i in range(self.a.shape[0]):
        #Получение предыдущих скрытых состояний
            if i==0:
                h_prev=np.zeros(self.Whh.shape[0])
            else:
                h_prev=self.h[i-1,:]
            self.z[i,:]=sigm(np.matmul(self.a[i,:],self.Wzx)+np.matmul(self.Wzh,h_prev)+self.bz)
            self.r[i,:]=sigm(np.matmul(self.a[i,:],self.Wrx)+np.matmul(self.Wrh,h_prev)+self.br)#Получение reset гейта
            self.h_cand[i,:]=tanh(np.matmul(self.a[i,:],self.Whx)+np.matmul(self.Whh,(self.r[i,:]*h_prev))+self.bh)#Получение reset гейта#Получение скрытого-состояния - кандидата
            self.h[i,:]=(1-self.z[i,:])*h_prev+self.z[i,:]*self.h_cand[i,:]#Получение скрытого состояния
        self.outputs=np.matmul(self.h,self.Wy).reshape(-1)+self.by#Проход через полносвязный слой   
        
    def _compute_loss(self, X, y):
        y_pred = self.predict_2(X)
        return self.loss_fn(y_pred, y)


    def _backprop(self, y, batch_size=4):
        self.dWzh=np.zeros_like(self.Wzh)
        self.dWzx=np.zeros_like(self.Wzx)
        self.dWrh=np.zeros_like(self.Wrh)
        self.dWrx=np.zeros_like(self.Wrx)
        self.dWhh=np.zeros_like(self.Whh)
        self.dWhx=np.zeros_like(self.Whx)
        self.dWy=np.zeros_like(self.Wy)
        self.dbz = np.zeros_like(self.bz)
        self.dbr = np.zeros_like(self.br)
        self.dbh = np.zeros_like(self.bh)
        self.dby=np.zeros_like(self.by)
        #Обратный проход через полносвязный слой
        doutput=de(self.outputs,y)[0]
        self.dWy = np.matmul(doutput, self.h).reshape(-1,1)
        self.dby = np.sum(doutput)
        delta_z=np.zeros_like(self.h)
        delta_r=np.zeros_like(self.h)
        delta_h=np.zeros_like(self.h)
        #Проход через реккурентный слой (BPTT)
        for i in range(batch_size-1,-1,-1):
            if i==batch_size-1:
                delta_z_next=np.zeros(self.Whh.shape[0])
                delta_r_next=np.zeros(self.Whh.shape[0])
                delta_h_next=np.zeros(self.Whh.shape[0])
            else:
                delta_z_next=delta_z[i+1,:]
                delta_r_next=delta_r[i+1,:]
                delta_h_next=delta_h[i+1,:]
            delta_h[i,:]=dtanh(self.h[i,:])*((doutput[i]*self.Wy.T)[0]*(1-self.z[i,:])+np.matmul(delta_h_next,self.Whh.T))
            delta_z[i,:]=dsigm(self.z[i,:])*((doutput[i]*self.Wy.T)[0]*self.z[i,:]*(self.h_cand[i-1,:]-self.h[i,:])+np.matmul(delta_z_next,self.Wzh.T))
            delta_r[i,:]=dsigm(self.r[i,:])*(np.matmul((doutput[i]*self.Wy.T)[0]*(1-self.z[i,:])*dtanh(self.h_cand[i,:]),self.Whh)*self.h[i-1,:]+np.matmul(delta_r_next,self.Wrh.T))
            #Получение дельт на весах
        for i in range(1,batch_size):
            self.dWzh+=np.matmul(delta_z[i,:].reshape(-1,1),self.h[i-1,:].reshape(1,-1))
            self.dWzx+=np.matmul(self.a[i-1,:].reshape(-1,1),delta_z[i,:].reshape(1,-1))
            self.dWrh+=np.matmul(delta_r[i,:].reshape(-1,1),self.h[i-1,:].reshape(1,-1))
            self.dWrx+=np.matmul(self.a[i-1,:].reshape(-1,1),delta_r[i,:].reshape(1,-1))
            self.dWhh+=np.matmul(delta_h[i,:].reshape(-1,1),(self.h[i-1,:]*self.r[i,:]).reshape(1,-1))
            self.dWhx+=np.matmul(self.a[i-1,:].reshape(-1,1),delta_h[i,:].reshape(1,-1))
            self.dbh+= delta_h[i,:]
            self.dbz+= delta_z[i,:]
            self.dbr+= delta_r[i,:]

  
    def _update_params_sgd(self, lr=1e-2): 
        self.Wzh-=lr*self.dWzh
        self.Wzx-=lr*self.dWzx
        self.Wrh-=lr*self.dWrh
        self.Wrx-=lr*self.dWrx
        self.Whh-=lr*self.dWhh
        self.Whx-=lr*self.dWhx
        self.Wy-=lr*self.dWy
        self.bz -=lr*self.dbz
        self.br -=lr*self.dbr
        self.bh -=lr*self.dbh
        self.by-=lr*self.dby
        
    def _update_params(self, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-7):
      # update first moments
        self.mWzh = beta_1*self.mWzh + (1-beta_1)*self.dWzh
        self.mWzx = beta_1*self.mWzx + (1-beta_1)*self.dWzx
        self.mWrh = beta_1*self.mWrh + (1-beta_1)*self.dWrh
        self.mWrx = beta_1*self.mWrx + (1-beta_1)*self.dWrx
        self.mWhh = beta_1*self.mWhh + (1-beta_1)*self.dWhh
        self.mWhx = beta_1*self.mWhx + (1-beta_1)*self.dWhx
        self.mWy = beta_1*self.mWy + (1-beta_1)*self.dWy
        self.mbr = beta_1*self.mbr + (1-beta_1)*self.dbr
        self.mbz = beta_1*self.mbz + (1-beta_1)*self.dbz
        self.mbh = beta_1*self.mbh + (1-beta_1)*self.dbh
        self.mby = beta_1*self.mby + (1-beta_1)*self.dby
      # update second moments
        self.vWzh = beta_2*self.vWzh + (1-beta_2)*(self.dWzh**2)
        self.vWzx = beta_2*self.vWzx + (1-beta_2)*(self.dWzx**2)
        self.vWrh = beta_2*self.vWrh + (1-beta_2)*(self.dWrh**2)
        self.vWhh = beta_2*self.vWhh + (1-beta_2)*(self.dWhh**2)
        self.vWrx = beta_2*self.vWrx + (1-beta_2)*(self.dWrx**2)
        self.vWhx = beta_2*self.vWhx + (1-beta_2)*(self.dWhx**2)
        self.vWy = beta_2*self.vWy + (1-beta_2)*(self.dWy**2)
        self.vbz = beta_2*self.vbz + (1-beta_2)*(self.dbz**2)
        self.vbr = beta_2*self.vbr + (1-beta_2)*(self.dbr**2)
        self.vbh = beta_2*self.vbh + (1-beta_2)*(self.dbh**2)
        self.vby = beta_2*self.vby + (1-beta_2)*(self.dby**2)
      # correction
        mWzh_corr = self.mWzh / (1-beta_1**self.t)
        mWzx_corr = self.mWzx / (1-beta_1**self.t)
        mWrh_corr = self.mWrh / (1-beta_1**self.t)
        mWhh_corr = self.mWhh / (1-beta_1**self.t)
        mWrx_corr = self.mWrx / (1-beta_1**self.t)
        mWhx_corr = self.mWhx / (1-beta_1**self.t)
        mWy_corr = self.mWy / (1-beta_1**self.t)
        mbr_corr = self.mbr / (1-beta_1**self.t)
        mbz_corr = self.mbz / (1-beta_1**self.t)
        mbh_corr = self.mbh / (1-beta_1**self.t)
        mby_corr = self.mby / (1-beta_1**self.t)
        vWzh_corr = self.vWzh / (1-beta_2**self.t)
        vWzx_corr = self.vWzx / (1-beta_2**self.t)
        vWrh_corr = self.vWrh / (1-beta_2**self.t)
        vWrx_corr = self.vWrx / (1-beta_2**self.t)
        vWhh_corr = self.vWhh / (1-beta_2**self.t)
        vWhx_corr = self.vWhx / (1-beta_2**self.t)
        vWy_corr = self.vWy / (1-beta_2**self.t)
        vbz_corr = self.vbz / (1-beta_2**self.t)
        vbr_corr = self.vbr / (1-beta_2**self.t)
        vbh_corr = self.vbh / (1-beta_2**self.t)
        vby_corr = self.vby / (1-beta_2**self.t)
        
      # update params
        self.Wzh -= lr*mWzh_corr / (np.sqrt(vWzh_corr)+eps)
        self.Wzx -= lr*mWzx_corr / (np.sqrt(vWzx_corr)+eps)
        self.Wrh -= lr*mWrh_corr / (np.sqrt(vWrh_corr)+eps)
        self.Whh -= lr*mWhh_corr / (np.sqrt(vWhh_corr)+eps)
        self.Wrx -= lr*mWrx_corr / (np.sqrt(vWrx_corr)+eps)
        self.Whx -= lr*mWhx_corr / (np.sqrt(vWhx_corr)+eps)
        self.Wy -= lr*mWy_corr / (np.sqrt(vWy_corr)+eps)
        self.br -= lr*mbr_corr / (np.sqrt(vbr_corr)+eps)
        self.bz -= lr*mbz_corr / (np.sqrt(vbz_corr)+eps)
        self.bh -= lr*mbh_corr / (np.sqrt(vbh_corr)+eps)
        self.by -= lr*mby_corr / (np.sqrt(vby_corr)+eps)
        
    # update time
        self.t += 1


    def train(self, X, Y, epochs=1, batch_size=4, lr=0.0001, adam=False):
        #Разбиение на мини-батчи
        x=X.reshape(int(X.shape[0]/batch_size),batch_size,X.shape[1])
        y=Y.reshape(-1,batch_size)
        epoch_losses = np.array([])
        dataset = list(zip(x, y))
        for i in range(epochs):
            rng.shuffle(dataset)
            for (x_b, y_b) in dataset:
                self._feedforward(x_b)
                self._backprop(y_b)
                if adam==False:
                    self._update_params_sgd(lr=lr)
                else:
                    self._update_params(lr=lr)
            epoch_losses = np.append(epoch_losses, self._compute_loss(X, Y))
        return epoch_losses


    def predict(self, x):#Предсказание на всём ряду
        a=x
        h_prev=np.zeros(self.Whh.shape[0])
        predicts=np.array([])
        for i in range(a.shape[0]):
            z=sigm(np.matmul(a[i,:],self.Wzx)+np.matmul(self.Wzh,h_prev)+self.bz)
            r=sigm(np.matmul(a[i,:],self.Wrx)+np.matmul(self.Wrh,h_prev)+self.br)
            h_cand=tanh(np.matmul(a[i,:],self.Whx)+np.matmul(self.Whh,(r*h_prev).reshape(-1))+self.bh)
            h=(1-z)*h_prev+z*h_cand
            h_prev=h.reshape(-1)
            pred=(np.matmul(h,self.Wy)+self.by).reshape(-1)
            predicts=np.append(predicts, pred)
        return predicts   

    def predict_2(self, X_b, b_size=4):#Предсказание на мини-батчах
        X=X_b.reshape(int(X_b.shape[0]/b_size),b_size,X_b.shape[1])
        predicts=np.array([])
        for x in X:
            a=x
            h_prev=np.zeros(self.Whh.shape[0])
            for i in range(a.shape[0]):
                z=sigm(np.matmul(a[i,:],self.Wzx)+np.matmul(self.Wzh,h_prev)+self.bz)
                r=sigm(np.matmul(a[i,:],self.Wrx)+np.matmul(self.Wrh,h_prev)+self.br)
                h_cand=tanh(np.matmul(a[i,:],self.Whx)+np.matmul(self.Whh,(r*h_prev).reshape(-1))+self.bh)
                h=(1-z)*h_prev+z*h_cand
                h_prev=h.reshape(-1)
                pred=(np.matmul(h,self.Wy)+self.by).reshape(-1)  
                predicts=np.append(predicts,pred)
        return predicts