import matplotlib.pyplot as plt
import numpy as np
import warnings
from utils.activations import softmax, d_softmax, relu, relu2deriv, d_softmax_cross_entropy, softmax_last
from utils.metrics import cross_entropy_loss
rng = np.random.default_rng(51)
warnings.filterwarnings("ignore")

def dynamic_padding(X, n_heads,n_mlp_neurons):#Динамический пэддинг для батча
    pad_max=max([x.shape[0] for x in X])#Находим максимальный размер предложения, по которому осуществляем padding
    vec_size=X[0].shape[1]#Длина эмбеддинга
    pad_mask=np.zeros((len(X),pad_max,vec_size))#Маска пэддинга для начальных операций
    pad_mask_2=np.ones((len(X),n_heads,pad_max,pad_max))*(-10**7) #Маска пэддинга для софтмакса (если будут 0, то softmax неправильно отработает)
    pad_mask_mlp=np.zeros((len(X), pad_max, n_mlp_neurons))#Маска пэддинга для прохода через 2 слой MLP
    for i,x in enumerate(X):#Пэддинг и создание масок пэддинга для всего батча
        pad_mask[i,:x.shape[0],:]+=x
        pad_mask_2[i,:,:x.shape[0],:x.shape[0]]=0
        pad_mask_mlp[i,:x.shape[0],:]=1
    return pad_mask[:,None,:,:], pad_mask_2,pad_mask_mlp


def get_batches(data, batch_size):
    n = len(data)
    get_X = lambda z: z[0]
    get_y = lambda z: z[1]
    for i in range(0, n, batch_size):
        batch = data[i:i+batch_size]
        yield [get_X(b) for b in batch], np.array([get_y(b) for b in batch])
        

class BatchNormalizator: #Класс батч-нормализатора
    def __init__(self, size):
        self.gamma=np.random.normal(0,1,size=size)/10
        self.beta=np.random.normal(0,1,size=size)/10
        self.dgamma=np.zeros_like(self.gamma)
        self.dbeta=np.zeros_like(self.dgamma)
        self.mgamma=np.zeros_like(self.dgamma)
        self.mbeta=np.zeros_like(self.dgamma)
        self.vgamma=np.zeros_like(self.dgamma)
        self.vbeta=np.zeros_like(self.dgamma)
        self.t = 1
    def feedforward(self, X, pad_mask, eps=10**-7):#Прямой проход
        self.eps=eps
        batch_mean=np.sum(np.sum(X,axis=1),axis=0)/np.sum(np.count_nonzero(X,axis=1),axis=0)[0] #Средние с учётом пэддинга
        self.X_mu=(X-batch_mean)*(pad_mask!=0)
        square_sum=np.square(X-batch_mean)*(pad_mask!=0)#Сумма квадратов отклонений с учётом пэддинга
        self.batch_sigm=np.sum(np.sum(square_sum,axis=1),axis=0)/np.sum(np.count_nonzero(square_sum,axis=1),axis=0)[0] #Дисперсия
        self.sqrt_sigm=np.sqrt(self.batch_sigm+self.eps)
        self.X_hat=(X-batch_mean)/self.sqrt_sigm#Подсчёт нормализованных иксов
        y_hat=(self.X_hat*self.gamma+self.beta)*(pad_mask!=0) #Выход с учётом пэддинга
        return y_hat  
    def backprop(self, delta, pad_mask):#Обратный проход
        self.dbeta=np.sum(np.sum(delta,axis=1),axis=0)
        self.dgamma=np.sum(np.sum(self.X_hat*delta,axis=1),axis=0)
        delta=delta*self.gamma
        d_X_mu_1=delta/self.sqrt_sigm
        d_ivar=np.sum(np.sum(self.X_mu*delta,axis=1),axis=0)
        d_sqrt_sigm=-1/(self.sqrt_sigm**2)*d_ivar
        d_batch_sigm= 0.5/np.sqrt(self.batch_sigm+self.eps) * d_sqrt_sigm
        d_sq=d_batch_sigm*(np.ones_like(self.X_mu)/np.sum(np.count_nonzero(self.X_mu,axis=1),axis=0)[0])
        d_X_mu_2=2*self.X_mu*d_sq
        d_x1=d_X_mu_2+d_X_mu_1
        dmu=-np.sum(np.sum(d_x1,axis=1),axis=0)
        d_x2=dmu*(np.ones_like(self.X_mu)/np.sum(np.count_nonzero(self.X_mu,axis=1),axis=0)[0])
        dx=(d_x1+d_x2)*pad_mask
        return dx
    def gradient_step(self, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-7):#Обновление весов по adam-у
        self.mgamma= beta_1*self.mgamma + (1-beta_1)*self.dgamma
        self.mbeta = beta_1*self.mbeta + (1-beta_1)*self.dbeta
        self.vgamma = beta_2*self.vgamma+ (1-beta_2)*(self.dgamma**2)
        self.vbeta = beta_2*self.vbeta + (1-beta_2)*(self.dbeta**2)
        mgamma_corr = self.mgamma    / (1-beta_1**self.t)
        mbeta_corr = self.mbeta    / (1-beta_1**self.t)
        vgamma_corr = self.vgamma    / (1-beta_2**self.t)
        vbeta_corr = self.vbeta    / (1-beta_2**self.t)
        self.gamma -= lr*mgamma_corr / (np.sqrt(vgamma_corr)+eps)
        self.beta -= lr*mbeta_corr / (np.sqrt(vbeta_corr)+eps)
        self.t += 1

    def normalize(self,X, pad_mask):#Нормализация
        batch_mean=np.sum(np.sum(X,axis=1),axis=0)/np.sum(np.count_nonzero(X,axis=1),axis=0)[0] #Средние с учётом пэддинга
        square_sum=np.square(X-batch_mean)*(pad_mask!=0)#Сумма квадратов отклонений с учётом пэддинга
        batch_sigm=np.sum(np.sum(square_sum,axis=1),axis=0)/np.sum(np.count_nonzero(square_sum,axis=1),axis=0)[0] #Дисперсия
        sqrt_sigm=np.sqrt(batch_sigm+self.eps)
        X_hat=(X-batch_mean)/sqrt_sigm#Подсчёт нормализованных иксов
        y_hat=(X_hat*self.gamma+self.beta)*(pad_mask!=0) #Выход с учётом пэддинга
        return y_hat
    
class Transformer:
    def __init__(self, n_heads, n_attention_out, n_mlp_neurons, embedding_len):
        self.activation_fn = relu
        self.activation_dfn = relu2deriv
        self.loss_fn = cross_entropy_loss
        self.loss_dfn = d_softmax_cross_entropy
        self.n_heads=n_heads #Число голов у attention-а
        self.n_mlp_neurons=n_mlp_neurons #Число нейронов на скрытом слое MLP
        self.WQ, self.WK,self.WV,self.WO = self._init_weights_attention(self.n_heads, n_attention_out,embedding_len)#Инициализация весов attention и слоя на выходе attention
        self.W_mlp=self._init_weights((embedding_len,n_mlp_neurons,embedding_len))#Инициализация весов для MLP
        self.b_mlp=self._init_biases((embedding_len,n_mlp_neurons, embedding_len))#Инициализация смещений для MLP
        self.W_last=self._init_weights((embedding_len,2))#Инициализация весов для выходного слоя
        self.b_last=self._init_biases((embedding_len,2))#Инициализация смещений для выходного слоя
        #Градиенты весов
        self.dWQ=np.zeros_like(self.WQ)
        self.dWK=np.zeros_like(self.WK)
        self.dWV=np.zeros_like(self.WV)
        self.dWO=np.zeros_like(self.WO)
        self.dW_mlp=[np.zeros_like(self.W_mlp[i]) for i in range(len(self.W_mlp))]
        self.db_mlp=[np.zeros_like(self.b_mlp[i]) for i in range(len(self.b_mlp))]
        self.dW_last=[np.zeros_like(self.W_last[i]) for i in range(len(self.W_last))]
        self.db_last=[np.zeros_like(self.b_last[i]) for i in range(len(self.b_last))]
        # Adam optimizer params
        self.t = 1
        self.mWQ=np.zeros_like(self.WQ)
        self.mWK=np.zeros_like(self.WK)
        self.mWV=np.zeros_like(self.WV)
        self.mWO=np.zeros_like(self.WO)
        self.vWQ=np.zeros_like(self.WQ)
        self.vWK=np.zeros_like(self.WK)
        self.vWV=np.zeros_like(self.WV)
        self.vWO=np.zeros_like(self.WO)
        self.mW_mlp = [np.zeros_like(w) for w in self.W_mlp]
        self.mb_mlp = [np.zeros_like(b) for b in self.b_mlp]
        self.vW_mlp = [np.zeros_like(w) for w in self.W_mlp]
        self.vb_mlp = [np.zeros_like(w) for w in self.b_mlp]
        self.mW_last = [np.zeros_like(w) for w in self.W_last]
        self.mb_last = [np.zeros_like(b) for b in self.b_last]
        self.vW_last = [np.zeros_like(w) for w in self.W_last]
        self.vb_last = [np.zeros_like(w) for w in self.b_last]
  
  
  # Функция инициализации весов для MLP
    def _init_weights(self, arch):
        net_in = arch[0]
        net_out = arch[-1]
        limit = np.sqrt(6. / (net_in + net_out))
        return [rng.uniform(-limit, limit + 1e-5, size=(arch[i], arch[i+1])) for i in range(len(arch)-1)]
    # Функция инициализации весов для attention
    def _init_weights_attention(self, n_heads,n_attention_out,embedding_len):
        limit = np.sqrt(6. / (n_attention_out+200))
        WQ=rng.uniform(-limit, limit + 1e-5, size=(n_heads,embedding_len,n_attention_out))
        WK=rng.uniform(-limit, limit + 1e-5, size=(n_heads,embedding_len,n_attention_out))
        WV=rng.uniform(-limit, limit + 1e-5, size=(n_heads,embedding_len,n_attention_out))
        WO=rng.uniform(-limit, limit + 1e-5, size=(n_heads*n_attention_out,embedding_len))
        return WQ, WK, WV, WO

# Функция инициализации смещений для MLP
    def _init_biases(self, arch):
        return [rng.random((1,arch[i+1]))*2-1 for i in range(len(arch)-1)]

#Прямой проход
    def feedforward(self, X, normalisator_1, normalisator_2):
        self.X_batch, self.pad_mask_2, self.pad_mask_mlp=dynamic_padding(X, n_heads=self.n_heads,n_mlp_neurons=self.n_mlp_neurons)#Создание пэддингов и масок для батча
        self.pad_mask=(self.X_batch!=0)[:,0,:,:]
        self.Q=np.matmul(self.X_batch,self.WQ)#Матрица запросов
        self.K=np.matmul(self.X_batch,self.WK)#Матриа ключей
        self.V=np.matmul(self.X_batch,self.WV)#Матрица значений
        QK=np.matmul(self.Q, self.K.transpose(0,1,3,2)) / np.sqrt(self.Q.shape[-1])
        SM=softmax(QK+self.pad_mask_2)#Софтмакс от произведения матриц запросов и ключей
        self.SM_2 = np.nan_to_num(SM, nan=0.0)#Заменяем nan-ы на 0 (так сделано специально, чтобы при софтмаксе пэддинги ни на что не повлияли)
        self.attention=np.matmul(self.SM_2,self.V)#Итоговый выход блока внимания
        Z=np.matmul(self.attention.transpose(0,2,1,3).reshape(self.attention.shape[0],self.attention.shape[2],-1),self.WO)#Выпрямленный и отмасштабированный выход
        res_1=Z+self.X_batch[:,0,:,:] #Добавленный вход в attention
        self.X_batch_normed=normalisator_1.feedforward(res_1,self.pad_mask)#Батч-нормализация
        #Проход через 2 полносвязных слоя
        self.z_1=(np.matmul(self.X_batch_normed,self.W_mlp[0])+self.b_mlp[0])*self.pad_mask_mlp
        self.a_1=relu(self.z_1)
        self.z_2=(np.matmul(self.a_1,self.W_mlp[1])+self.b_mlp[1])*(self.pad_mask!=0)
        self.a_2=self.z_2
        res_2=self.X_batch_normed+self.a_2#Добавка остаточного блока
        self.X_batch_normed_2=normalisator_2.feedforward(res_2,self.pad_mask)#Батч-нормализация
        self.GAP_X=np.sum(self.X_batch_normed_2,axis=1)/np.count_nonzero(self.X_batch_normed_2,axis=1)#Global Average Pooling
        self.z_last=np.matmul(self.GAP_X,self.W_last[0])+self.b_last[0] 
        self.a_last=softmax_last(self.z_last)#Итоговый результат
        


    def _compute_loss(self, X, y, normalisator_1, normalisator_2):
        y_pred = self.predict(X, normalisator_1, normalisator_2)
        pred=np.argmax(y_pred,axis=1)#Строгий предикт
        return self.loss_fn(y_pred, y), np.sum(pred==y)/len(y)


    def backprop(self, y, normalisator_1, normalisator_2, batch_size=32):
        delta=d_softmax_cross_entropy(self.a_last,y)
        #Обновление весов и смещения последнего линейного слоя
        self.dW_last[-1] = np.matmul(self.GAP_X.T, delta)
        self.db_last[-1] = np.sum(delta, axis=0, keepdims=True)
        delta=np.matmul(delta, self.W_last[-1].T)
        #Обратный проход через global average pooling
        delta=np.repeat(delta/np.count_nonzero(self.X_batch_normed_2,axis=1),self.X_batch_normed_2.shape[1],axis=0).reshape(self.X_batch_normed_2.shape)*(self.pad_mask!=0)
        delta=normalisator_2.backprop(delta, self.pad_mask)#Проход через batch_norm
        delta_2=delta
        #Обратный проход через полносвязные слои
        self.dW_mlp[-1]=np.sum(np.matmul(self.a_1.transpose(0,2,1),delta),axis=0)
        self.db_mlp[-1]=np.sum(np.sum(delta,axis=0),axis=0)
        delta=np.matmul(delta,self.W_mlp[-1].T)*relu2deriv(self.z_1)
        self.dW_mlp[0]=np.sum(np.matmul(self.X_batch_normed.transpose(0,2,1),delta),axis=0)
        self.db_mlp[0]=np.sum(np.sum(delta,axis=0),axis=0)
        delta=np.matmul(delta,self.W_mlp[0].T)
        #Добавление дельты до остаточного блока
        delta+=delta_2
        #Обратный проход по batch_norm
        delta=normalisator_1.backprop(delta, self.pad_mask)
        #Обратный проход через полносвязный слой
        self.dWO=np.sum(np.matmul(self.attention.transpose(0,2,1,3).reshape(self.attention.shape[0],self.attention.shape[2],-1).transpose(0,2,1),
                                  delta),axis=0)
        #Delta по выходу из attention
        delta=np.matmul(delta,self.WO.T).reshape(delta.shape[0],delta.shape[1],self.n_heads,-1).transpose(0,2,1,3)
        #Backprop по attention
        delta_v=np.matmul(self.SM_2.transpose(0,1,3,2),delta)#Дельта по values
        self.dWV=np.sum(np.matmul(self.X_batch.transpose(0,1,3,2),delta_v),axis=0)
        delta_s=np.matmul(delta,self.V.transpose(0,1,3,2))#Дельта по софтмаксу
        delta=(np.matmul(delta_s.reshape(delta_s.shape[0],delta_s.shape[1],delta_s.shape[2],1,delta_s.shape[3]).astype('float32'),np.apply_along_axis(d_softmax,3,self.SM_2.astype('float32')).astype('float32'))[:,:,:,0,:]*(self.pad_mask_2==0))
        delta_q=np.matmul(delta,self.K)/np.sqrt(self.X_batch.shape[-1])
        delta_k=np.matmul(self.Q.transpose(0,1,3,2),delta).transpose(0,1,3,2)/np.sqrt(self.X_batch.shape[-1])
        self.dWK=np.sum(np.matmul(self.X_batch.transpose(0,1,3,2),delta_k),axis=0)
        self.dWQ=np.sum(np.matmul(self.X_batch.transpose(0,1,3,2),delta_q),axis=0)


  # adam 
    def _update_params(self, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-7):
        for i in range(len(self.W_mlp)):
          # update first moments
            self.mW_mlp[i] = beta_1*self.mW_mlp[i] + (1-beta_1)*self.dW_mlp[i]
            self.mb_mlp[i] = beta_1*self.mb_mlp[i] + (1-beta_1)*self.db_mlp[i]
              # update second moments
            self.vW_mlp[i] = beta_2*self.vW_mlp[i] + (1-beta_2)*(self.dW_mlp[i]**2)
            self.vb_mlp[i] = beta_2*self.vb_mlp[i] + (1-beta_2)*(self.db_mlp[i]**2)
              # correction
            mW_corr = self.mW_mlp[i] / (1-beta_1**self.t)
            mb_corr = self.mb_mlp[i] / (1-beta_1**self.t)
            vW_corr = self.vW_mlp[i] / (1-beta_2**self.t)
            vb_corr = self.vb_mlp[i] / (1-beta_2**self.t)
              # update params
            self.W_mlp[i] -= lr*mW_corr / (np.sqrt(vW_corr)+eps)
            self.b_mlp[i] -= lr*mb_corr / (np.sqrt(vb_corr)+eps)
        for i in range(len(self.W_last)):
          # update first moments
            self.mW_last[i] = beta_1*self.mW_last[i] + (1-beta_1)*self.dW_last[i]
            self.mb_last[i] = beta_1*self.mb_last[i] + (1-beta_1)*self.db_last[i]
              # update second moments
            self.vW_last[i] = beta_2*self.vW_last[i] + (1-beta_2)*(self.dW_last[i]**2)
            self.vb_last[i] = beta_2*self.vb_last[i] + (1-beta_2)*(self.db_last[i]**2)
              # correction
            mW_corr = self.mW_last[i] / (1-beta_1**self.t)
            mb_corr = self.mb_last[i] / (1-beta_1**self.t)
            vW_corr = self.vW_last[i] / (1-beta_2**self.t)
            vb_corr = self.vb_last[i] / (1-beta_2**self.t)
              # update params
            self.W_last[i] -= lr*mW_corr / (np.sqrt(vW_corr)+eps)
            self.b_last[i] -= lr*mb_corr / (np.sqrt(vb_corr)+eps)
        # update time
        self.mWQ = beta_1*self.mWQ + (1-beta_1)*self.dWQ
        self.mWK = beta_1*self.mWK + (1-beta_1)*self.dWK
        self.mWV = beta_1*self.mWV+ (1-beta_1)*self.dWV
        self.mWO = beta_1*self.mWO + (1-beta_1)*self.dWO
          # update second moments
        self.vWQ = beta_2*self.vWQ + (1-beta_2)*(self.dWQ**2)
        self.vWK = beta_2*self.vWK + (1-beta_2)*(self.dWK**2)
        self.vWV = beta_2*self.vWV + (1-beta_2)*(self.dWV**2)
        self.vWO = beta_2*self.vWO + (1-beta_2)*(self.dWO**2)
          # correction
        mWQ_corr = self.mWQ    / (1-beta_1**self.t)
        mWK_corr = self.mWK    / (1-beta_1**self.t)
        mWV_corr = self.mWV    / (1-beta_1**self.t)
        mWO_corr = self.mWO    / (1-beta_1**self.t)
        vWQ_corr = self.vWQ    / (1-beta_2**self.t)
        vWK_corr = self.vWK    / (1-beta_2**self.t)
        vWV_corr = self.vWV    / (1-beta_2**self.t)
        vWO_corr = self.vWO    / (1-beta_2**self.t)
          # update params
        self.WQ -= lr*mWQ_corr / (np.sqrt(vWQ_corr)+eps)
        self.WK -= lr*mWK_corr / (np.sqrt(vWK_corr)+eps)
        self.WV -= lr*mWV_corr / (np.sqrt(vWV_corr)+eps)
        self.WO -= lr*mWO_corr / (np.sqrt(vWO_corr)+eps)
        self.t += 1

    #Тренировка
    def train(self, X, y, X_t, y_t, normalisator_1, normalisator_2, epochs=1, batch_size=32):
        epoch_losses = np.array([])
        dataset = list(zip(X, y))
        self.lr_start=0.00003
        for i in range(epochs):
            rng.shuffle(dataset)
            for (X_batch, y_batch) in get_batches(dataset, batch_size):
                self.feedforward(X_batch, normalisator_1, normalisator_2)
                self.backprop(y_batch, normalisator_1, normalisator_2)
                self._update_params(lr=self.lr_start)
                normalisator_1.gradient_step(lr=self.lr_start)#Вместе с весами сети обновляем и батч-нормализаторы
                normalisator_2.gradient_step(lr=self.lr_start)
            self.lr_start=self.lr_start*0.99 #Экспоненциальное замедление learning rate
            losses=self._compute_loss(X_t, y_t, normalisator_1, normalisator_2)
            epoch_losses = np.append(epoch_losses, losses[0])
            plt.plot(epoch_losses)
            plt.title(f"Accuracy:{losses[1]}")
            plt.show()#Каждую эпоху выводим график
        return epoch_losses


    def predict(self, X, normalisator_1, normalisator_2):#Предсказание вероятностей классов
        X_batch, pad_mask_2,pad_mask_mlp=dynamic_padding(X, n_heads=self.n_heads,n_mlp_neurons=self.n_mlp_neurons)
        pad_mask=(X_batch!=0)[:,0,:,:]
        Q=np.matmul(X_batch,self.WQ)#Матрица запросов
        K=np.matmul(X_batch,self.WK)#Матриа ключей
        V=np.matmul(X_batch,self.WV)#Матрица значений
        QK=np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(Q.shape[-1])
        SM=softmax(QK+pad_mask_2)#Софтмакс от произведения матриц запросов и ключей
        SM_2 = np.nan_to_num(SM, nan=0.0)#Заменяем nan-ы на 0 (так сделано специально, чтобы при софтмаксе пэддинги ни на что не повлияли)
        attention=np.matmul(SM_2,V)#Итоговый выход блока внимания
        Z=np.matmul(attention.transpose(0,2,1,3).reshape(attention.shape[0],attention.shape[2],-1),self.WO)#Выпрямленный и отмасштабированный выход
        res_1=Z+X_batch[:,0,:,:] #Добавленный вход в attention
        X_batch_normed=normalisator_1.normalize(res_1,pad_mask)#Батч-нормализация
        #Проход через 2 полносвязных слоя
        z_1=(np.matmul(X_batch_normed,self.W_mlp[0])+self.b_mlp[0])*pad_mask_mlp
        a_1=relu(z_1)
        z_2=(np.matmul(a_1,self.W_mlp[1])+self.b_mlp[1])*pad_mask
        #a_2=relu(z_2)
        a_2=z_2
        res_2=X_batch_normed+a_2#Добавка остаточного блока
        X_batch_normed_2=normalisator_2.normalize(res_2,pad_mask)#Батч-нормализация
        GAP_X=np.sum(X_batch_normed_2,axis=1)/np.count_nonzero(X_batch_normed_2,axis=1)#Global Average Pooling
        z_last=np.matmul(GAP_X,self.W_last[0])+self.b_last[0] 
        a_last=softmax_last(z_last)#Итоговый результат
        return a_last
    
    def strict_predict(self, X, normalisator_1, normalisator_2):#Строгое предсказание классов
        return np.argmax(self.predict(X, normalisator_1, normalisator_2), axis=1)
    


