import os
import random
import sys
import numpy as np
import scipy.io as sio
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.preprocessing import StandardScaler

from utils import util


def load_data(config):
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []

    if data_name in ['Scene_15']:
      
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name+'.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))


    elif data_name in ['hand']:
    
        mat = sio.loadmat(os.path.join(main_dir,'data', 'handwritten.mat'))
        X_data = mat['X'][0]
                 
        X_list.append(X_data[0].astype(np.float32))
        X_list.append(X_data[1].astype(np.float32))
        X_list.append(X_data[2].astype(np.float32))
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['LandUse_21']:


        scaler = StandardScaler()
        random.seed(config["seed"])
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse_21.mat'))
        train_x = []
        train_x.append(sparse.csr_matrix(mat['X'][0, 0]).toarray())  # 20
        train_x.append(sparse.csr_matrix(mat['X'][0, 1]).toarray())  # 59
        train_x.append(sparse.csr_matrix(mat['X'][0, 2]).toarray())  # 40
        index = random.sample(range(train_x[0].shape[0]), 2100)  # 30000
        for view in [1, 2]:
            print("view",view," ",np.shape(train_x[view][index]))                 
            x = scaler.fit_transform(train_x[view][index]).astype('float32')
            y = np.squeeze(mat['Y']).astype('int')[index]
            X_list.append(x)
            Y_list.append(y)


    elif data_name in ['NoisyMNIST']:

        data = sio.loadmat('./data/NoisyMNIST.mat')
        X1 = data['X1'].astype('float32')  # conversion float32 pour PyTorch
        X2 = data['X2'].astype('float32')
        Y  = data['Y'].squeeze()           # vecteur 1D

        # Créer un objet DataSet unique
        full_dataset = DataSet_NoisyMNIST(X1, X2, Y)

        # Construire les listes pour CAIMVR
        X_list.append(full_dataset.images1)
        X_list.append(full_dataset.images2)
        Y_list.append(full_dataset.labels)
        Y_list.append(full_dataset.labels)


    elif data_name in ['Caltech101-7']:

        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        # print(mat)        X = mat['X']
        Y = np.squeeze(mat['Y']).astype('int')

        for view in [3, 4]:
            x = X[view]

            if isinstance(x, np.ndarray) and x.shape == (1,):
                x = x[0]
            # --------------------------------------------------------------
            x = util.normalize(x).astype('float32')
            X_list.append(x)
            Y_list.append(Y)

    elif data_name in ['MSRC_v1']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        
        X_data = mat['X']
        num_views = X_data.shape[1]
        
        print(f"Num_views: {num_views}")
        
        X_list = []
        
        selected_views = [2, 3]  
        
        for idx in selected_views:
            X = X_data[0, idx].astype(np.float32)
            
            scaler = MinMaxScaler()
            X_norm = scaler.fit_transform(X)
            
            X_list.append(X_norm)
        
        Y = np.squeeze(mat['Y']).astype(np.int32)
        Y_list = [Y for _ in range(len(selected_views))]

    elif data_name in ['Caltech101-20']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        X = mat['X'][0]
        for view in [3, 4]:
            x = X[view]
            x = util.normalize(x).astype('float32')
            y = np.squeeze(mat['Y']).astype('int')
            X_list.append(x)
            Y_list.append(y)

    return X_list, Y_list

