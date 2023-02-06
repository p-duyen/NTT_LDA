import numpy as np
from matplotlib import pyplot as plt

def comp_mean_vectors(X, y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    mean_vectors = []
    for cl in class_labels:
        mean_vectors.append(np.mean(X[y==cl], axis=0))
    return mean_vectors

def scatter_within(X, y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)
    S_W = np.zeros((n_features, n_features))
    for cl, mv in zip(class_labels, mean_vectors):
        class_sc_mat = np.zeros((n_features, n_features))
        for row in X[y == cl]:
            row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat
    return S_W

def scatter_between(X, y):
    overall_mean = np.mean(X, axis=0)
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)
    S_B = np.zeros((n_features, n_features))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(n_features, 1)
        overall_mean = overall_mean.reshape(n_features, 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return S_B

def get_components(eig_vals, eig_vecs, n_comp=2):
    n_features = X.shape[1]
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.hstack([eig_pairs[i][1].reshape(4, 1) for i in range(0, n_comp)])
    return W
def gen_data_m(states, n_random, n_samples, q, noise=0.1):
    data = {}
    dim = states.shape[1]
    labels = np.concatenate((np.ones((n_samples + n_samples%2)//2, dtype=np.int16), np.zeros((n_samples-(n_samples + n_samples%2)//2), dtype=np.int16)), axis=0)
    np.random.shuffle(labels)
    sts = states[labels]
    x0 = np.random.randint(0, q, size=(n_samples, dim))
    x1 = (sts - x0)%q
    L0 = HW(x0) + np.random.normal(0, noise, (n_samples, dim))
    L1 = HW(x1) + np.random.normal(0, noise, (n_samples, dim))
    # L = np.concatenate((L0, L1), axis=1)
    L = L0*L1
    L = np.hstack([L, np.random.randn(n_samples, n_random)])
    data['labels'] = labels
    data['traces'] = L
    return data
def gen_data(states, n_random, n_samples, noise=0.1):
    """
    L = HW(key)||random
    """
    data = {}
    # labels = np.random.randint(0, states.shape[0], (n_samples, ))
    labels = np.concatenate((np.ones((n_samples + n_samples%2)//2, dtype=np.int16), np.zeros((n_samples-(n_samples + n_samples%2)//2), dtype=np.int16)), axis=0)
    np.random.shuffle(labels)
    X = states[labels]
    X = X + np.random.normal(0, noise, (n_samples, 1))
    X = np.hstack([X, np.random.randn(n_samples, n_random)])
    data['labels'] = labels
    data['traces'] = X
    return data

def HW(x):
    fbin = np.vectorize(np.binary_repr)
    bin_x = [fbin(xi) for xi in x]
    fcount = np.vectorize(np.char.count)
    hw = np.array([fcount(xi, '1') for xi in bin_x])
    return hw
if __name__ == '__main__':
    states = np.array([[1], [3]])
    states = HW(states)
    # p_data = gen_data(states, n_random=3, n_samples=1000, noise=0.1)
    p_data = gen_data_m(states, n_random=3,  n_samples=10000, q=5)
    X = p_data['traces']
    y = p_data['labels']
    means = comp_mean_vectors(X, y)
    print(means)
    Sw = scatter_within(X, y)
    Sb = scatter_between(X, y)
    print(Sw)
    print(Sb)
    print(np.linalg.inv(Sw).dot(Sb))
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    print('EigVals: %s\n\nEigVecs: %s' % (eig_vals, eig_vecs))
    W = get_components(eig_vals, eig_vecs, n_comp=1)
    print(W.shape)
    X_lda = X.dot(W)
    print("X", X_lda[y==0, 0].shape)
    Y = np.arange(5000)
    # plt.scatter(Y, X_lda[y==0, 0], color = 'hotpink')
    # plt.scatter(Y, X_lda[y==1, 0], color = '#88c999')
    # for label,marker,color in zip(np.unique(y),('^', 's'),('blue', 'red')):
    #     print(label)
        # plt.scatter(X_lda[y==label, 0], X_lda[y==label, 1],
        #         color=color, marker=marker)
    # plt.show()
    for label,marker,color in zip(np.unique(y),('^', 's'),('blue', 'red')):
        plt.scatter(Y, X_lda[y==label, 0], color = color, marker=marker)
    plt.show()
