import numpy as np
from utils.metrics import bce, dbce
from utils.activations import sigm, dsigm
rng = np.random.default_rng(51)

def get_batches(data, batch_size):
    n = len(data)
    get_X = lambda z: z[0]
    get_y = lambda z: z[1]
    for i in range(0, n, batch_size):
        batch = data[i:i+batch_size]
        yield np.array([get_X(b) for b in batch]), np.array([get_y(b) for b in batch])


class MLP:
    def __init__(self, architecture):
        self.depth = len(architecture)-1
 
        self.activation_fn = sigm 
        self.activation_dfn = dsigm 
        self.loss_fn = bce 
        self.loss_dfn = dbce 
        # Инициализация весов
        self.W = self._init_weights(architecture)
        self.b = self._init_biases(architecture)
        # Выходы слоёв
        self.z = [None] * (self.depth+1)
        self.a = [None] * (self.depth+1)
        # Градиенты
        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b]
        # Оптимизаторы Адама
        self.t = 1
        self.mW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(w) for w in self.b]

  
  # Инициализация весов через glorot
    def _init_weights(self, arch):
        net_in = arch[0]
        net_out = arch[-1]
        limit = np.sqrt(6. / (net_in + net_out))
        return [rng.uniform(-limit, limit + 1e-5, size=(arch[i+1], arch[i])) for i in range(self.depth)]

  #Инициализация смещений
    def _init_biases(self, arch):
        return [rng.random((arch[i+1],1))*2-1 for i in range(self.depth)]

  #Прямой проход
    def _feedforward(self, X):
        self.a[0] = X.T #На вход пускаем транспонированный батч
        for i in range(self.depth):
            self.z[i+1] = np.matmul(self.W[i], self.a[i]) + self.b[i]
            self.a[i+1] = self.activation_fn(self.z[i+1]) #Поскольку идёт задача классификации, 
                                                        #выходной слой формируется после функции активации

    def _compute_loss(self, X, y):
        y_pred = self.predict(X).reshape(y.shape)
        return self.loss_fn(y_pred, y)

    #Обратный поход
    def _backprop(self, y, batch_size=32):
        delta = self.loss_dfn(self.a[-1], y)
        for i in range(self.depth-1, 0, -1):
            if i != self.depth-1:
                delta = self.activation_dfn(self.z[i+1]) * np.matmul(self.W[i+1].T, delta)
            else:
                delta = self.loss_dfn(self.a[-1], y)*self.activation_dfn(self.z[-1])#На последнем скрытом слое сначала надо посчитать произведение 
                                                                                    #функции потерь и активации
            self.dW[i] = np.matmul(delta, self.a[i].T)
            self.db[i] = np.sum(delta, axis=1, keepdims=True)

  
    def _update_params_sgd(self, lr=1e-2): #Используем градиентный спуск
        for i in range(self.depth):
            self.W[i] -= lr*self.dW[i]
            self.b[i] -= lr*self.db[i]


    def train(self, X, y, epochs=1, batch_size=32):
        epoch_losses = np.array([])
        dataset = list(zip(X, y))
        for _ in range(epochs):
            rng.shuffle(dataset)
            for (X_batch, y_batch) in get_batches(dataset, batch_size):
                self._feedforward(X_batch)
                self._backprop(y_batch)
                self._update_params_sgd()
            epoch_losses = np.append(epoch_losses, self._compute_loss(X, y))
        return epoch_losses


    def predict(self, X):
        a = X.T #На вход пускаем транспонированный батч
       
        for i in range(self.depth): 
            a = np.matmul(self.W[i], a) + self.b[i]
            a = self.activation_fn(a) #Функцию активации применяем после всех слоёв
        return a
    
    def predict_classes(self, X, p_min=0.5): #Функция для расчёта значения по вероятности
        a = X.T
        # compute hidden and output layers
        for i in range(self.depth): 
            a = np.matmul(self.W[i], a) + self.b[i]
            a = self.activation_fn(a)
        a[a>p_min] = 1
        a[a<=p_min] = 0
        return a