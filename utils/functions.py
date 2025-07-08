import numpy as np

def im2col(X, kernel_size):#Функция, позволяющая развернуть элементы, по которым идёт проход ядром, в столбцы матрицы
    result = np.empty([X.shape[0],X.shape[1],kernel_size[0] * kernel_size[1], (X.shape[2] - kernel_size[0] + 1)*(X.shape[3] - kernel_size[1] + 1)])
    for i in range(X.shape[3] - kernel_size[1] + 1):
        for j in range(X.shape[2] - kernel_size[0] + 1):
            result[:,:,:, i *(X.shape[2] - kernel_size[0] + 1)+ j] = X[:,:,i:i + kernel_size[0], j:j + kernel_size[1]].reshape(X.shape[0],X.shape[1],kernel_size[0] * kernel_size[1])
    return result

def col2im(X, image_size, kernel_size):#Функция, обратно сворачивающая столбцы в картинки нужного размера
    result = np.zeros([X.shape[0],X.shape[1],image_size[0],image_size[1]])
    weight = np.zeros([X.shape[0],X.shape[1],image_size[0],image_size[1]])
    col = 0
    for i in range(image_size[1] - kernel_size[1] + 1):
        for j in range(image_size[0] - kernel_size[0] + 1):
            result[:,:,i:i + kernel_size[0], j:j + kernel_size[1]] += X[:,:,:, col].reshape(X.shape[0],X.shape[1],kernel_size[0],kernel_size[1])
            weight[:,:,i:i + kernel_size[0], j:j + kernel_size[1]] += np.ones(kernel_size)
            col += 1
    return result / weight