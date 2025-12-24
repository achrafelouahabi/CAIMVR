import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


def vote(lsd1, lsd2, label, n=1):
    """Sometimes the prediction accuracy will be higher in this way.
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :param n: Similar to K in k-nearest neighbors algorithm
    :return: Predicted label
    """
    F_h_h = np.dot(lsd2, np.transpose(lsd1))
    gt_list = []
    label = label.reshape(len(label), 1)
    for num in range(n):
        F_h_h_argmax = np.argmax(F_h_h, axis=1)
        F_h_h_onehot = convert_to_one_hot(F_h_h_argmax, len(label))
        F_h_h = F_h_h - np.multiply(F_h_h, F_h_h_onehot)
        gt_list.append(np.dot(F_h_h_onehot, label))
    gt_ = np.array(gt_list).transpose(2, 1, 0)[0].astype(np.int64)
    count_list = []
    count_list.append([np.argmax(np.bincount(gt_[i])) for i in range(lsd2.shape[0])])
    gt_pre = np.array(count_list)
    return gt_pre.transpose()


def ave(lsd1, lsd2, label):
    """Classification using nearest neighbor with cosine similarity
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :return: Predicted label
    """
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    knn.fit(lsd1, label)
    return knn.predict(lsd2)


def ave_with_proba(lsd1, lsd2, label, n_neighbors=5):
    """Classification with probability scores for AUC calculation
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :param n_neighbors: number of neighbors for KNN
    :return: Predicted labels and probability scores
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(lsd1, label)
    
    label_pre = knn.predict(lsd2)
    proba = knn.predict_proba(lsd2)
    
    return label_pre, proba


def ave_with_scores(lsd1, lsd2, label):
    """Classification with similarity scores (alternative for AUC)
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :return: Predicted labels and similarity scores
    """
    # Calculer la similarité cosinus
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity(lsd2, lsd1)
    
    # Vérifier si labels commencent à 0 ou 1
    label_min = label.min()
    labels_normalized = label - label_min  # Normaliser à 0
    
    # Pour chaque échantillon test, trouver le label du plus proche voisin
    nearest_indices = np.argmax(similarities, axis=1)
    label_pre = label[nearest_indices]
    
    # Calculer les scores moyens par classe
    n_classes = len(np.unique(label))
    scores = np.zeros((lsd2.shape[0], n_classes))
    
    for i in range(n_classes):
        class_label = i + label_min
        class_mask = (label == class_label)
        if np.any(class_mask):
            # Score moyen pour cette classe
            scores[:, i] = np.mean(similarities[:, class_mask], axis=1)
    
    return label_pre, scores


def ave_original(lsd1, lsd2, label):
    """Original average method (kept for reference)
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :return: Predicted label
    """
    F_h_h = np.dot(lsd2, np.transpose(lsd1))
    
    # Vérifier si les labels commencent à 0 ou 1
    label_min = label.min()
    
    # Si labels commencent à 0
    if label_min == 0:
        label = label.reshape(len(label), 1)
        enc = OneHotEncoder()
        a = enc.fit_transform(label)
        label_onehot = a.toarray()
        label_num = np.sum(label_onehot, axis=0)
        F_h_h_sum = np.dot(F_h_h, label_onehot)
        F_h_h_mean = F_h_h_sum / label_num
        label_pre = np.argmax(F_h_h_mean, axis=1)
    
    # Si labels commencent à 1
    else:
        label = label.reshape(len(label), 1) - 1
        enc = OneHotEncoder()
        a = enc.fit_transform(label)
        label_onehot = a.toarray()
        label_num = np.sum(label_onehot, axis=0)
        F_h_h_sum = np.dot(F_h_h, label_onehot)
        F_h_h_mean = F_h_h_sum / label_num
        label_pre = np.argmax(F_h_h_mean, axis=1) + 1
    
    return label_pre


def ave_original_with_proba(lsd1, lsd2, label):
    """Original average method with probability scores
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :return: Predicted labels and probability scores
    """
    F_h_h = np.dot(lsd2, np.transpose(lsd1))
    
    # Vérifier si les labels commencent à 0 ou 1
    label_min = label.min()
    
    # Si labels commencent à 0
    if label_min == 0:
        label = label.reshape(len(label), 1)
        enc = OneHotEncoder()
        a = enc.fit_transform(label)
        label_onehot = a.toarray()
        label_num = np.sum(label_onehot, axis=0)
        F_h_h_sum = np.dot(F_h_h, label_onehot)
        F_h_h_mean = F_h_h_sum / label_num
        
        # Normaliser pour obtenir des pseudo-probabilités
        from scipy.special import softmax
        proba = softmax(F_h_h_mean, axis=1)
        
        label_pre = np.argmax(F_h_h_mean, axis=1)
    
    # Si labels commencent à 1
    else:
        label = label.reshape(len(label), 1) - 1
        enc = OneHotEncoder()
        a = enc.fit_transform(label)
        label_onehot = a.toarray()
        label_num = np.sum(label_onehot, axis=0)
        F_h_h_sum = np.dot(F_h_h, label_onehot)
        F_h_h_mean = F_h_h_sum / label_num
        
        # Normaliser pour obtenir des pseudo-probabilités
        from scipy.special import softmax
        proba = softmax(F_h_h_mean, axis=1)
        
        label_pre = np.argmax(F_h_h_mean, axis=1) + 1
    
    return label_pre, proba


