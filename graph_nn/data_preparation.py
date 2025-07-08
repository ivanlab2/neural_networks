import json
import pandas as pd
import numpy as np
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.analysis.dimensionality import find_connected_atoms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

rng = np.random.default_rng(51)

def read_pymatgen_dict(file):#Чтение JSON-а
    with open(file, "r") as f:
        d = json.load(f)
    return Structure.from_dict(d)
    
def prepare_dataset(dataset_path):#Подготовка тренировочного и тестового датасета
    dataset_path = Path(dataset_path)
    targets = pd.read_csv(dataset_path / "targets.csv", index_col=0)
    struct = {
        item.name.strip(".json"): read_pymatgen_dict(item)
        for item in (dataset_path / "structures").iterdir()
    }

    data = pd.DataFrame(columns=["structures"], index=struct.keys())
    data = data.assign(structures=struct.values(), targets=targets)

    return train_test_split(data, test_size=0.25, random_state=666)

def find_avg_bonds(struct):#Функция по нахождению средней длины ребра атома и её стандартизация
    bonds=np.array([])
    for j in range(len(struct)):
        dist=0
        for i in range(len(struct)):
            if i!=j:
                dist+=struct.get_distance(j,i)
        bonds=np.append(bonds,dist/191)
    return (bonds-np.min(bonds))/(np.max(bonds)-np.min(bonds))    

def prepare_data(dataset):#Функция по подготовке датасета с нужными признаками, матрицами смежности и реальными размерами молекул
    for i in range(dataset.shape[0]):
        struct=dataset['structures'].iloc[i]#Берём i-ую молекулу
        adj_matrix=find_connected_atoms(struct,tolerance=5)#Находим матрицу смежности
        adj_matrix=adj_matrix+np.eye(adj_matrix.shape[0])#Добавляем единичную матрицу для учёта данных о признаке в графовой свёртке
        species= np.array([site.species for site in struct.sites])#Находим названия атомов для молекулы
        if i==0:
            ohe=OneHotEncoder(sparse_output=False)
            ohe.fit(species)
        transformed=ohe.transform(species)#Преобразуем атомы в помощью one-hot экнодинга
        d=np.concatenate((transformed,find_avg_bonds(struct).reshape(-1,1)),axis=1)#Конкатенируем метки атомов и средние длины рёбер
        d=np.pad(d, ((0, 192-d.shape[0]), (0, 0)), mode='constant', constant_values=0) #Пэддинг матрицы признаков 
        real_shape=adj_matrix.shape[0]#Реальные размеры молекул (нужны для корректного проведения global average pooling)
        adj_matrix=np.pad(adj_matrix, ((0, 192-adj_matrix.shape[0]), (0, 192-adj_matrix.shape[1])))#Пэддинг матрицы смежности
        #Склеивание данных о молекуле с данными о других молекулах
        if i==0:
            data=d[None,:,:]
            matrices=adj_matrix[None,:,:]
            real_shapes=np.array([real_shape])
        else:
            data=np.concatenate((data,d[None,:,:]),axis=0)
            matrices=np.concatenate((matrices,adj_matrix[None,:,:]),axis=0)
            real_shapes=np.append(real_shapes,real_shape)
    return data, matrices, real_shapes