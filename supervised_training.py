import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import f1_score
from scipy import stats


# preprocessing_types = ['flat', 'mean_channels', 'pca', 'kernel_pca']
preprocessing_types = ['flat']
#model_types = ['LR', 'RF', 'LDA', 'KNN']
model_types = ['RF']

def TSNE_plot(X, y):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE()
    X_flat = preprocess_flat(X)
    embeddings = tsne.fit_transform(X_flat)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y)
    plt.savefig("tsne_plot.png")


def preprocess_flat(X):
    X_flat = []
    for x in X:
        X_flat.append(x.flatten())
    X_flat = np.array(X_flat)
    return X_flat

def preprocess_mean_channels(X):
    X_mean = np.mean(X, 1)
    X_mean = np.mean(X_mean, 1)
    X_mean = np.mean(X_mean, 1)
    
    X_std = np.std(X, 1)
    X_std = np.std(X_std, 1)
    X_std = np.std(X_std, 1)
    
    X_mean_flat = preprocess_flat(X_mean)
    X_std_flat = preprocess_flat(X_std)
    X_preprocessed = np.hstack([X_mean_flat, X_std_flat])
    
    return X_preprocessed

def preprocess_mean_voxels(X):
    X_mean = np.mean(X, -1)
    X_std = np.std(X, -1)
    X_mean_flat = preprocess_flat(X_mean)
    X_std_flat = preprocess_flat(X_std)
    X_preprocessed = np.hstack([X_mean_flat, X_std_flat])
    return X_preprocessed
    
def preprocess_pca(X, type='linear'):
    if type=='linear':
        from sklearn.decomposition import PCA
        pca = PCA(0.9)
    elif type=='kernel':
        print("HERE")
        from sklearn.decomposition import KernelPCA
        pca = KernelPCA(150, kernel='sigmoid')

    X_flat = preprocess_flat(X)
    print(np.shape(X_flat))
    X_reduced = pca.fit_transform(X_flat)
    return X_reduced

def preprocess_nmf(X):
    from sklearn.decomposition import NMF
    nmf = NMF(90, tol=1e-2)

    X_flat = preprocess_flat(X)
    
    X_flat = X_flat - np.min(X_flat)
    print(np.shape(X_flat))
    X_reduced = nmf.fit_transform(X_flat)
    print("Reconstruction Error: ", nmf.reconstruction_err_)
    return X_reduced

def get_accuracy(preprocessing, model_type):
    X_train = np.load("supervised/X_train.npy")
    Y_train = np.load("supervised/Y_train.npy")
    X_train = np.squeeze(X_train, axis=1)
    X_test = np.load("supervised/X_test.npy")
    Y_test = np.load("supervised/Y_test.npy")
    X_test = np.squeeze(X_test, axis=1)
    train_len = len(X_train)
    X = np.vstack([X_train, X_test])
    if preprocessing == 'flat':
        X = preprocess_flat(X)
    elif preprocessing == 'mean_channels':
        X = preprocess_mean_channels(X)
    elif preprocessing == 'mean_voxels':
        X = preprocess_mean_voxels(X)
    elif preprocessing == 'pca':
        X = preprocess_pca(X)
    elif preprocessing == 'kernel_pca':
        X = preprocess_pca(X, 'kernel')
    elif preprocessing == 'nmf':
        X = preprocess_nmf(X)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    X_train = X[:train_len]
    X_test = X[train_len:]
    TSNE_plot(X_train, Y_train)
    # exit()
    if model_type == 'LR':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1)
    elif model_type == 'SVM':
        from sklearn.svm import SVC
        model = SVC(kernel='rbf')
    elif model_type == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(max_features=0.1, n_estimators=500, max_depth=4)
    elif model_type == 'LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()
    elif model_type == 'MLP':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier((128, 128), alpha=1, activation='logistic')
    elif model_type == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=9)

    # print(model)
    model.fit(X_train, Y_train)
    train_accuracy = model.score(X_train, Y_train)
    test_accuracy = model.score(X_test, Y_test)
    y_pred = model.predict(X_test)
    Y_test = np.reshape(Y_test, [len(Y_test)//10, 10])
    y_pred = np.reshape(y_pred, [len(y_pred)//10, 10])
    Y_test = stats.mode(Y_test, axis=1)[0].flatten()
    y_pred = stats.mode(y_pred, axis=1)[0].flatten()    
    test_accuracy = np.mean(Y_test == y_pred)
    return train_accuracy, test_accuracy


results = {}
for model_type in model_types:
    results[model_type] = {}
    for preprocessing in preprocessing_types:
        train_accuracy, test_accuracy = get_accuracy(preprocessing, model_type)
        results[model_type][preprocessing] = (train_accuracy, test_accuracy)
        print("Preprocessing: {}, Model: {}, Train Acc: {}, Test Acc:{}".format(preprocessing,
                                                                               model_type,
                                                                               train_accuracy,
                                                                               test_accuracy))
        with open("Results2.pkl", 'wb') as f:
            pickle.dump(results, f)

for k in results:
    for v in results[k]:
        print(k, v, results[k][v])
with open("Results2.pkl", 'wb') as f:
    pickle.dump(results, f)
