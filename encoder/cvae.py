import matplotlib.pyplot as plt
import numpy as np
from utils.activations import relu, relu2deriv, sigm, dsigm
from utils.metrics import mse, dmse, KLD, dKLD
from utils.functions import im2col, col2im
rng = np.random.default_rng(51)


def get_batches(data, batch_size):
    n = len(data)
    get_X = lambda z: z[0]
    get_y = lambda z: z[1]
    for i in range(0, n, batch_size):
        batch = data[i:i+batch_size]
        yield np.array([get_X(b) for b in batch]), np.array([get_y(b) for b in batch])

class CVAE:
    def __init__(self, arch_enc, latent_size, arch_decod):
        self.depth_enc = len(arch_enc) #Число слоёв энкодера
        self.depth_dec = len(arch_decod) #Число слоёв декодера
        #Функции активации и потерь
        self.activation_fn = relu
        self.activation_dfn = relu2deriv
        self.loss_dec = mse #loss декодера - разница в пикселях между реальныи изображением и сгенерированным 
        self.loss_d_dec = dmse 
        self.beta=0.001 #Уменьшение влияния KLD на градиенты
        # Параметры модели
        self.W_enc = self._init_weights_enc(arch_enc, latent_size)
        self.b_enc = self._init_biases_enc(arch_enc, latent_size)
        self.W_dec = self._init_weights_dec(arch_decod, latent_size)
        self.b_dec = self._init_biases_dec(arch_decod, latent_size)
        self.res_enc=[None]*(self.depth_enc)#Результаты функции im2col
        self.z_enc = [None] * (self.depth_enc)
        self.z_dec = [None] * (self.depth_dec+1)
        self.a_dec = [None] * (self.depth_dec+1)
        self.m_enc=[None] * (self.depth_enc+1)#Входные значения слоёв
        self.a_enc = [None] * (self.depth_enc)
        self.masks_enc=[None]*(self.depth_enc)#Маски макспулинга
        self.eps=np.array([])
        self.sigma=np.array([])
        # backprop gradients
        self.dW_enc = [np.zeros_like(w) for w in self.W_enc]
        self.db_enc = [np.zeros_like(b) for b in self.b_enc]
        self.dW_dec = [np.zeros_like(w) for w in self.W_dec]
        self.db_dec = [np.zeros_like(b) for b in self.b_dec]
        # Adam optimizer params
        self.t_vae = 1
        self.mW_enc = [np.zeros_like(w) for w in self.W_enc]
        self.mb_enc = [np.zeros_like(b) for b in self.b_enc]
        self.vW_enc = [np.zeros_like(w) for w in self.W_enc]
        self.vb_enc = [np.zeros_like(b) for b in self.b_enc]
        self.mW_dec = [np.zeros_like(w) for w in self.W_dec]
        self.mb_dec = [np.zeros_like(b) for b in self.b_dec]
        self.vW_dec = [np.zeros_like(w) for w in self.W_dec]
        self.vb_dec = [np.zeros_like(b) for b in self.b_dec]



    def _init_weights_enc(self, arch, latent_size): #Инициализация весов энкодера
        W=[]
        s=np.sqrt(arch[0])
        net_in = arch[0]
        net_out = latent_size
        limit = np.sqrt(6. / (net_in + net_out))
        for i in range(1,self.depth_enc+1):
            if i!=self.depth_enc: #Инициализация весов для полносвязного слоя
                W.append(np.array(rng.uniform(-limit, limit + 1e-5, size=(4,arch[i],arch[i]))))
                s=int((s-arch[i]+1))
            else:#Инициализация весов для свёрточных слоёв
                w=[]
                w.append(np.array(rng.uniform(-limit, limit + 1e-5, size=(latent_size,4*np.square(s)))))
                w.append(np.array(rng.uniform(-limit, limit + 1e-5, size=(latent_size,4*np.square(s)))))
                W.append(w)
        return W
        

    def _init_biases_enc(self, arch, latent_size):#Инициализация смещений энкодера
        b=[]
        for i in range(1,self.depth_enc+1):
            if i!=self.depth_enc: 
                b.append(rng.random((4,1,1)))#Инициализация смещений для полносвязного слоя
            else:
                bb=[]
                bb.append(rng.random((latent_size,1)))
                bb.append(rng.random((latent_size,1)))
                b.append(bb)#Инициализация смещений для свёрточных слоёв
        return b
        
    def _init_weights_dec(self, arch, latent_size):#Инициализация весов декодера
        arch=[latent_size+10]+arch
        net_in = arch[0]
        net_out = arch[-1]
        limit = np.sqrt(6. / (net_in + net_out))
        return [rng.uniform(-limit, limit + 1e-5, size=(arch[i+1], arch[i])) for i in range(self.depth_dec)]

    def _init_biases_dec(self, arch, latent_size):#Инициализация смещений декодера
        arch=[latent_size+10]+arch
        return [rng.random((arch[i+1],1))*2-1 for i in range(self.depth_dec)]    

    def feedforward_encoder(self, X): #Прямой проход энкодера
        self.m_enc[0]=np.repeat(X,4,axis=1).reshape(X.shape[0],4,X.shape[1],X.shape[2],order='F')
        for i in range(self.depth_enc-1):#Проход через свёрточные слои
            self.res_enc[i]=im2col(self.m_enc[i],(self.W_enc[i].shape[1],self.W_enc[i].shape[1]))
            self.a_enc[i]=(np.matmul(self.W_enc[i].reshape(self.W_enc[i].shape[0],1,-1), self.res_enc[i])+self.b_enc[i]).reshape(self.m_enc[i].shape[0],self.m_enc[i].shape[1],-1)
            self.z_enc[i]=self.activation_fn(self.a_enc[i])  
            self.m_enc[i+1]=self.z_enc[i].reshape(self.z_enc[i].shape[0],self.z_enc[i].shape[1],np.sqrt(self.z_enc[i].shape[2]).astype('int'),np.sqrt(self.z_enc[i].shape[2]).astype('int'))
        #Создания среднего и дисперсии
        self.m_enc[self.depth_enc-1]=self.m_enc[self.depth_enc-1].reshape(self.m_enc[self.depth_enc-1].shape[0],-1)
        self.z_enc[self.depth_enc-1] = [np.matmul(self.m_enc[self.depth_enc-1], self.W_enc[self.depth_enc-1][0].T)+self.b_enc[-1][0].T,
                         np.matmul(self.m_enc[self.depth_enc-1], self.W_enc[self.depth_enc-1][1].T)+self.b_enc[-1][1].T]
        return self.z_enc[self.depth_enc-1]
        
    def reparametrize(self,mu, sigma): #Трюк с репараметризацией
        self.eps=np.random.normal(size=mu.shape)
        self.sigma=sigma
        return mu+np.exp(0.5*self.sigma)*self.eps

    def d_reparametrize(self,delta):#Пропуск градиента через репараметризацию
        d_mu=delta
        d_sigma=delta*0.5*np.exp(0.5*self.sigma)*self.eps
        return [d_mu,d_sigma]
        
    def feedforward_decoder(self, X,y): #Прямой проход декодера (обычный MLP)
        self.a_dec[0] = np.concatenate((X.T,y.T)) 
        for i in range(self.depth_dec):
            self.z_dec[i+1] = np.matmul(self.W_dec[i], self.a_dec[i]) + self.b_dec[i]
            if i!=self.depth_dec-1:
                self.a_dec[i+1] = relu(self.z_dec[i+1])
            else:
                self.a_dec[i+1] = sigm(self.z_dec[i+1]) 
        return self.a_dec[-1]

    def feedforward_vae(self,X,y): #Прямой проход автоэнкодера
        mu,sigma=self.feedforward_encoder(X) #Прямой проход энкодера
        reparams=self.reparametrize(mu,sigma)#Репараметризация
        output=self.feedforward_decoder(reparams,y)#Прямой проход декодера
        return output
        
    def _compute_loss(self, X, y):#Подсчёт KLD и MSE от реальной картинки
        y_pred = self.predict(X,y)
        mu,sigma=self.predict_enc(X)
        return self.loss_dec(y_pred, X.reshape(X.shape[0],-1)), KLD(mu,sigma)

    def backprop_dec(self, y,batch_size=32):#Обратный проход энкодера без дискриминатора
        y=y.reshape(y.shape[0],-1).T
        delta = self.loss_d_dec(self.a_dec[-1], y)*dsigm(self.z_dec[-1])
        for i in range(self.depth_dec-1, -1, -1):
            if i != self.depth_dec-1:
                delta = self.activation_dfn(self.z_dec[i+1]) * np.matmul(self.W_dec[i+1].T, delta)
            else:
                delta = self.loss_d_dec(self.a_dec[-1], y)*dsigm(self.z_dec[-1])
            self.dW_dec[i] = np.matmul(delta, self.a_dec[i].T)
            self.db_dec[i] = np.sum(delta, axis=1, keepdims=True)
        return np.matmul(self.W_dec[0].T,delta).T[:,:-10]


    def backprop_enc(self, d_mu,d_sigma, batch_size=32):
        loss=dKLD(self.z_enc[self.depth_enc-1][0],self.z_enc[self.depth_enc-1][1])
        d_mu+=self.beta*loss[0]
        d_sigma+=self.beta*loss[1]
        #Проход через полносвязный слой
        self.dW_enc[-1][0] = np.matmul(d_mu.T, self.m_enc[self.depth_enc-1])
        self.db_enc[-1][0] = np.sum(d_mu, axis=0, keepdims=True).T
        self.dW_enc[-1][1] = np.matmul(d_sigma.T, self.m_enc[self.depth_enc-1])
        self.db_enc[-1][1] = np.sum(d_sigma, axis=0, keepdims=True).T
        delta=np.matmul(d_mu, self.W_enc[-1][0]) + np.matmul(d_sigma, self.W_enc[-1][1])
        #Превращение дельты в результат последней свёртки
        delta=np.reshape(delta, (delta.shape[0], 4,-1))
        for i in range(self.depth_enc-2,-1,-1):
            delta=delta*self.activation_dfn(self.z_enc[i])#Проход через слой активации
            self.db_enc[i]=np.mean(np.sum(delta,axis=2),axis=0).reshape(4,1,1)#Получение дельт смещений
            self.dW_enc[i]=np.mean(np.matmul(delta.reshape(delta.shape[0],delta.shape[1],1,-1),np.transpose(self.res_enc[i], (0,1, 3, 2))).reshape(delta.shape[0],delta.shape[1],-1),axis=0).reshape(self.W_enc[i].shape[0],self.W_enc[i].shape[1],self.W_enc[i].shape[2])#Обновление весов
            delta=col2im(np.matmul(self.W_enc[i].reshape(self.W_enc[i].shape[0],-1,1),delta.reshape(delta.shape[0],delta.shape[1],1,-1)),(self.m_enc[i].shape[2],self.m_enc[i].shape[2]),(self.W_enc[i].shape[1],self.W_enc[i].shape[1]))
            delta=np.reshape(delta, (delta.shape[0], 4,-1))

    def backprop_vae(self,y):#Обратный проход автоэнкодера без подключения дискриминатора
        delta=self.backprop_dec(y)
        d_mu,d_sigma=self.d_reparametrize(delta)
        self.backprop_enc(d_mu,d_sigma)
        
    def _update_params_sgd_vae(self, lr=1e-2):
        for i in range(self.depth_enc):
            self.W_enc[i] -= lr*self.dW_enc[i]
            self.b_enc[i] -= lr*self.db_enc[i]    
        for i in range(self.depth_enc):
            self.W_dec[i] -= lr*self.dW_dec[i]
            self.b_dec[i] -= lr*self.db_dec[i] 

    def _update_params_vae(self, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-7):
        for i in range(self.depth_enc):
      # update first moments
            self.mW_enc[i] = beta_1*self.mW_enc[i] + (1-beta_1)*self.dW_enc[i]
            self.mb_enc[i] = beta_1*self.mb_enc[i] + (1-beta_1)*self.db_enc[i]
      # update second moments
            self.vW_enc[i] = beta_2*self.vW_enc[i] + (1-beta_2)*(self.dW_enc[i]**2)
            self.vb_enc[i] = beta_2*self.vb_enc[i] + (1-beta_2)*(self.db_enc[i]**2)
      # correction
            mW_enc_corr = self.mW_enc[i] / (1-beta_1**self.t_vae)
            mb_enc_corr = self.mb_enc[i] / (1-beta_1**self.t_vae)
            vW_enc_corr = self.vW_enc[i] / (1-beta_2**self.t_vae)
            vb_enc_corr = self.vb_enc[i] / (1-beta_2**self.t_vae)
      # update params
            self.W_enc[i] -= lr*mW_enc_corr / (np.sqrt(vW_enc_corr)+eps)
            self.b_enc[i] -= lr*mb_enc_corr / (np.sqrt(vb_enc_corr)+eps)
    # update time
        
        for i in range(self.depth_dec):
          # update first moments
          self.mW_dec[i] = beta_1*self.mW_dec[i] + (1-beta_1)*self.dW_dec[i]
          self.mb_dec[i] = beta_1*self.mb_dec[i] + (1-beta_1)*self.db_dec[i]
          # update second moments
          self.vW_dec[i] = beta_2*self.vW_dec[i] + (1-beta_2)*(self.dW_dec[i]**2)
          self.vb_dec[i] = beta_2*self.vb_dec[i] + (1-beta_2)*(self.db_dec[i]**2)
          # correction
          mW_dec_corr = self.mW_dec[i] / (1-beta_1**self.t_vae)
          mb_dec_corr = self.mb_dec[i] / (1-beta_1**self.t_vae)
          vW_dec_corr = self.vW_dec[i] / (1-beta_2**self.t_vae)
          vb_dec_corr = self.vb_dec[i] / (1-beta_2**self.t_vae)
          # update params
          self.W_dec[i] -= lr*mW_dec_corr / (np.sqrt(vW_dec_corr)+eps)
          self.b_dec[i] -= lr*mb_dec_corr / (np.sqrt(vb_dec_corr)+eps)
        # update time
        self.t_vae += 1   

 
    def train(self, X,y, epochs=30, batch_size=32):
        epoch_losses_mse = np.array([])
        epoch_losses_kld = np.array([])
        dataset = list(zip(X,y))
        for i in range(epochs):#Сначала обучаем несколько эпох только автоэнкодер
            rng.shuffle(dataset)
            for (X_batch,y_batch) in get_batches(dataset, batch_size):
                self.feedforward_vae(X_batch,y_batch)
                self.backprop_vae(X_batch)
                self._update_params_vae(lr=1e-4)
            losses=self._compute_loss(X[:5000], y[:5000])
            epoch_losses_mse = np.append(epoch_losses_mse, losses[0])
            epoch_losses_kld = np.append(epoch_losses_kld, losses[1])
            plt.plot(epoch_losses_mse)
            plt.show()
 
        return epoch_losses_mse, epoch_losses_kld

    def predict(self, X,y): #Предсказание всего автоэнкодера
        mu,sigma=self.predict_enc(X)
        params=self.reparametrize(mu,sigma)
        values=self.predict_dec(params,y).T
        return values

    def predict_enc(self, X): #Предсказание энкодера
        t=np.repeat(X,4,axis=1).reshape(X.shape[0],4,X.shape[1],X.shape[2],order='F')
        for i in range(self.depth_enc-1):
            res=im2col(t,(self.W_enc[i].shape[1],self.W_enc[i].shape[1]))
            a=(np.matmul(self.W_enc[i].reshape(self.W_enc[i].shape[0],1,-1), res)+self.b_enc[i]).reshape(t.shape[0],t.shape[1],-1)
            z=self.activation_fn(a)
            t=z.reshape(z.shape[0],z.shape[1],np.sqrt(z.shape[2]).astype('int'),np.sqrt(z.shape[2]).astype('int'))
        t=t.reshape(t.shape[0],-1)
        z = [np.matmul(t, self.W_enc[self.depth_enc-1][0].T)+self.b_enc[-1][0].T,np.matmul(t, self.W_enc[self.depth_enc-1][1].T)+self.b_enc[-1][1].T]
        return z

    def predict_dec(self,X,y):#Предсказание декодера
        a = np.concatenate((X.T,y.T))
        for i in range(self.depth_dec): 
            a = np.matmul(self.W_dec[i], a) + self.b_dec[i]
            if i!=self.depth_dec-1:
                a = relu(a)
            else:
                a=sigm(a)
        return a
    def generate_digits(self, num_samples, hidden_size, onehotencoder):# Генерация рандомных цифр
        z = np.random.normal(0, 1, size=(num_samples, hidden_size))
        generated_digits=onehotencoder.transform(np.random.randint(0, 10, size=num_samples).reshape(-1,1))
        y_gen=self.predict_dec(z, generated_digits).T
        return y_gen
