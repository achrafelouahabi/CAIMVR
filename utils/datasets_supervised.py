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
                 
        scaler = MinMaxScaler()
        X_list.append(scaler.fit_transform(X_data[0]).astype(np.float32))
        X_list.append(scaler.fit_transform(X_data[2]).astype(np.float32))
        X_list.append(scaler.fit_transform(X_data[1]).astype(np.float32))
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))
  
    elif data_name in ['3Sources']:

        mat = sio.loadmat(os.path.join(main_dir, 'data', '3Sources.mat'))
        X_raw = mat['X'] 
        y = np.squeeze(mat['y'])  
        X_list = []
        Y_list = []
        scaler = MinMaxScaler()
        for i in range(X_raw.shape[0]): 
            view = X_raw[i, 0].astype(np.float32)
            # si les données sont 2D matrices -> flatten
            view = view.reshape(view.shape[0], -1)

            # normalisation
            view = normalize(view).astype(np.float32)

            X_list.append(view)
            Y_list.append(y)


    elif data_name in ['BBCSport']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'BBCSport.mat'))
        print("mat keys:", mat.keys())

        X_raw = mat['X'] 
        y = np.squeeze(mat['gt'])        
        print(f"X_raw shape: {X_raw.shape}")
        print(f"y shape: {y.shape}")

        X_list = []
        Y_list = []

        scaler = MinMaxScaler()
        n_views = X_raw.shape[1]  
        
        for i in range(n_views):  
            sparse_view = X_raw[0, i]  
            if hasattr(sparse_view, 'toarray'):
                view = sparse_view.toarray().astype(np.float32)
            elif hasattr(sparse_view, 'todense'):
                view = np.array(sparse_view.todense()).astype(np.float32)
            else:
                view = np.array(sparse_view).astype(np.float32)
            
            X_list.append(view)
            Y_list.append(y)

        print(f"Number of views: {len(X_list)}")
        print(f"View shapes: {[v.shape for v in X_list]}")



    elif data_name in ['LandUse_21']:

        # scaler = MinMaxScaler()
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

    elif data_name in ['DHA', 'UWA30']:
        train_data = data_loader_HAR(data_name)
        train_data.read_train()
        train_data_x, train_data_y, test_data_x, test_data_y, label = train_data.get_data()
        X_list.append(np.concatenate([train_data_x, test_data_x], axis=0))
        X_list.append(np.concatenate([train_data_y, test_data_y], axis=0))
        Y_list.append(label)
        Y_list.append(label)

    elif data_name in ['Caltech101-7']:

        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        # print(mat)
        X = mat['X']
        Y = np.squeeze(mat['Y']).astype('int')

        for view in [3, 4]:
            x = X[view]
            # --- Correction : déballer la cellule MATLAB si nécessaire ---
            if isinstance(x, np.ndarray) and x.shape == (1,):
                x = x[0]
            # --------------------------------------------------------------
            x = util.normalize(x).astype('float32')
            X_list.append(x)
            Y_list.append(Y)

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



import numpy as np

class DataSet_NoisyMNIST(object):
    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=np.float32, normalize=True):
        """
        Dataset NoisyMNIST prêt pour Transformers.
        Prétraitement :
        - Conversion float32
        - Normalisation [0,1] puis (x - mean)/std
        - Shuffle automatique à chaque epoch
        """
        if dtype not in (np.uint8, np.float32):
            raise TypeError(f'Invalid image dtype {dtype}, expected uint8 or float32')

        self.one_hot = one_hot
        self._epochs_completed = 0
        self._index_in_epoch = 0

        # if fake_data:
        #     self._num_examples = 10000
        #     self._images1 = np.ones((self._num_examples, 28, 28), dtype=np.float32)
        #     self._images2 = np.ones((self._num_examples, 28, 28), dtype=np.float32)
        #     if one_hot:
        #         self._labels = np.tile([1] + [0]*9, (self._num_examples, 1))
        #     else:
        #         self._labels = np.zeros(self._num_examples, dtype=np.int64)
        #     return

        # Vérification des dimensions
        assert images1.shape[0] == labels.shape[0], "images1 et labels ont des tailles différentes"
        assert images2.shape[0] == labels.shape[0], "images2 et labels ont des tailles différentes"
        self._num_examples = images1.shape[0]

        # Conversion dtype float32 et normalisation
        if dtype == np.float32:
            images1 = images1.astype(np.float32)
            images2 = images2.astype(np.float32)

            images1 /= 255.0
            images2 /= 255.0

            # if normalize:
            #     # Valeurs classiques MNIST
            #     mean, std = 0.1307, 0.3081
            #     images1 = (images1 - mean) / std
            #     images2 = (images2 - mean) / std

        self._images1 = images1
        self._images2 = images2
        self._labels = labels

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Retourne le batch suivant."""
        if fake_data:
            fake_image = np.ones((28,28), dtype=np.float32)
            if self.one_hot:
                fake_label = [1] + [0]*9
            else:
                fake_label = 0
            return ([fake_image]*batch_size, [fake_image]*batch_size, [fake_label]*batch_size)

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Nouvelle époque : shuffle
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]


def load_NoisyMNIST():
    data = sio.loadmat('./data/NoisyMNIST.mat')

    train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])

    tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])

    test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])

    return train, tune, test


class data_loader_HAR:

    def __init__(self, database_name='DHA'):

        self.filename = database_name
        self.data_x = []  # training and testing RGB + depth feature
        self.data_label = []  # training and testing depth label
        self.train_data_x = []  # training depth feature
        self.train_data_y = []  # training RGB feature
        self.train_data_label = []  # training label
        self.test_data_x = []  # testing depth feature
        self.test_data_y = []  # testing RGB feature
        self.test_data_label = []  # testing label
        # self.train_data_xy = []  # training RGB + depth feature
        # self.test_data_xy = []  # testing RGB + depth feature
        self.cluster = 0

    def read_train(self):

        # Depth feature -> 110-dimension
        # RGB feature -> 3x2048 dimension
        feature_num1 = 110
        feature_num2 = 3 * 2048
        feature_num = feature_num1 + feature_num2
        num = 0

        # load .csv file for training
        f = open('data/' + self.filename + '_total_train.csv', 'r')
        for i in f:
            num += 1
            row1 = i.rstrip().split(',')[:-1]
            row = [float(x) for x in row1]
            self.data_x.append(row[0:feature_num])
            self.data_label.append(row[feature_num:])
            self.train_data_x.append(row[0:feature_num1])
            self.train_data_y.append(row[feature_num1:feature_num1 + feature_num2])
            # self.train_data_xy.append(row[0:feature_num1 + feature_num2])
            self.train_data_label.append(row[feature_num1 + feature_num2:])
        f.close()

        # load .csv file for training
        f = open('data/' + self.filename + '_total_test.csv', 'r')
        for i in f:
            num += 1
            row1 = i.rstrip().split(',')[:-1]
            row = [float(x) for x in row1]
            self.data_x.append(row[0:feature_num])
            self.data_label.append(row[feature_num:])
            self.test_data_x.append(row[0:feature_num1])
            self.test_data_y.append(row[feature_num1:feature_num1 + feature_num2])
            # self.test_data_xy.append(row[0:feature_num1 + feature_num2])
            self.test_data_label.append(row[feature_num1 + feature_num2:])
        f.close()

        # random split training and test data
        train_data_x, train_data_y, test_data_x, test_data_y, label = self.get_data()
        data_x = np.concatenate([train_data_x, test_data_x], axis=0)
        data_y = np.concatenate([train_data_y, test_data_y], axis=0)
        self.train_data_x, self.test_data_x, self.train_data_y, self.test_data_y, self.train_data_label, self.test_data_label = train_test_split(
            data_x, data_y, label, test_size=0.5)

        # got the sample number
        self.sample_total_num = len(self.data_x)
        self.sample_train_num = len(self.train_data_x)
        self.sample_test_num = len(self.test_data_x)
        print(self.sample_total_num)

        self.cluster = len(self.data_label[0])

    def get_data(self):
        train_data_x = np.array(self.train_data_x)
        train_data_y = np.array(self.train_data_y)
        test_data_x = np.array(self.test_data_x)
        test_data_y = np.array(self.test_data_y)

        label = np.concatenate([self.train_data_label, self.test_data_label], axis=0)
        # label_new = [np.argmax(one_hot) for one_hot in label]

        return train_data_x, train_data_y, test_data_x, test_data_y, label

    # randomly choose _batch_size RGB and depth feature in the training set
    def train_next_batch(self, _batch_size):
        xx = []  # training batch of depth features
        yy = []  # training batch of RGB features
        zz = []  # training batch of labels
        for sample_num in random.sample(range(self.sample_train_num), _batch_size):
            xx.append(self.train_data_x[sample_num])
            yy.append(self.train_data_y[sample_num])
            zz.append(self.train_data_label[sample_num])
        return yy, xx, zz

    # randomly choose _batch_size RGB and depth feature in the testing set
    def test_next_batch(self, _batch_size):
        xx = []  # testing batch of depth features
        yy = []  # testing batch of RGB features
        zz = []  # testing batch of labels
        for sample_num in random.sample(range(self.sample_test_num), _batch_size):
            xx.append(self.test_data_x[sample_num])
            yy.append(self.test_data_y[sample_num])
            zz.append(self.test_data_label[sample_num])
        return yy, xx, zz
