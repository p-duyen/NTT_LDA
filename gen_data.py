import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from scalib.modeling import LDAClassifier as LDA
from matplotlib import pyplot as plt
import pdb
from tqdm import tqdm, trange
import pickle
# pdb.set_trace()

def count_1(x):
    return int(x).bit_count()

def HW(x):
    fcount = np.vectorize(count_1)
    return fcount(x)

def gen_bm8(batch_size, states_pair, noise=0, n_random=0):
    labels = np.concatenate((np.ones((batch_size + batch_size%2)//2, dtype=np.uint16), np.zeros((batch_size-(batch_size + batch_size%2)//2), dtype=np.uint16)), axis=0)
    np.random.shuffle(labels)
    sts = states_pair[labels]
    dim = states_pair.shape[1]
    r = np.random.randint(0, 256, (batch_size, dim))
    msts = r^sts
    L0 = HW(r) + np.random.normal(0, noise, (batch_size, dim))
    L1 = HW(msts) + np.random.normal(0, noise, (batch_size, dim))
    # L0 = HW(r)
    # L1 = HW(msts)
    L = L0*L1*10
    L = np.hstack([L, np.random.randint(0, 3, (batch_size, n_random))])
    L = L.astype(np.int16)
    # [print(f"labels:{labels[i]}, state:{sts[i]}, rand: {r[i]}, masked: {msts[i]}, L0:{L0[i]}, L1:{L1[i]}, L: {L[i]} ") for i in range(batch_size)]
    return labels, L


def gen_bm(batch_size, states_pair, base=2**8):
    labels = np.concatenate((np.ones((batch_size + batch_size%2)//2, dtype=np.uint16), np.zeros((batch_size-(batch_size + batch_size%2)//2), dtype=np.uint16)), axis=0)
    np.random.shuffle(labels)
    sts = states_pair[labels]
    r = np.random.randint(0, 256, (batch_size, 1), dtype=np.uint16)
    msts = (r^sts)%base
    L0 = HW(r)
    L1 = HW(msts)
    L = (L0*L1)
    L = np.hstack([L, np.random.randint(0, 3, (batch_size, 2))])
    L = L.astype(np.int16)
    # [print(f"labels:{labels[i]}, state:{sts[i]}, rand: {r[i]}, masked: {msts[i]}, L0:{L0[i]}, L1:{L1[i]}, L: {L[i]} ") for i in range(batch_size)]
    return labels, L
def gen_am(batch_size, states_pair, q=3329, noise=0.5):
    labels = np.concatenate((np.ones((batch_size + batch_size%2)//2, dtype=np.uint16), np.zeros((batch_size-(batch_size + batch_size%2)//2), dtype=np.uint16)), axis=0)
    np.random.shuffle(labels)
    sts = states_pair[labels]
    dim = states_pair.shape[1]
    r = np.random.randint(0, q, (batch_size, dim), dtype=np.uint16)
    msts = (sts - r)%q
    L0 = HW(r) + np.random.normal(0, noise, (batch_size, dim))
    L1 = HW(msts) + np.random.normal(0, noise, (batch_size, dim))
    L = L0*L1
    # L = L.astype(np.int16)
    # [print(f"labels:{labels[i]}, state:{sts[i]}, rand: {r[i]}, masked: {msts[i]}, L0:{L0[i]}, L1:{L1[i]}, L: {L[i]} ") for i in range(batch_size)]
    return labels, L
def gen_states():
    state_range = np.arange(0, 256)
    states_set = []
    for i in range(256):
        for j in range(i+1, 256):
            states_set.append([state_range[i], state_range[j]])
    states_set = np.array(states_set)
    return states_set
def plot_(traces, labels, states_pair, alpha, des=""):
    dim = traces.shape[1]
    len = traces.shape[0]
    X0 = traces[labels == 0]
    X1 = traces[labels == 1]
    if dim == 1:
        plt.scatter(X0, np.ones((len//2, ))*0.2, label=f"state:{states_pair[0][0]} {des}", alpha=alpha)
        plt.scatter(X1, np.ones((len//2, ))*-0.2, label=f"state:{states_pair[1][0]} {des}", alpha=alpha)


    else:
        plt.scatter(X0[:, 0], X0[:, 1], color="red", label=f"state:{states_pair[0][0]} {des}", alpha=alpha)
        plt.scatter(X1[:, 0], X1[:, 1], color="blue", label=f"state:{states_pair[1][0]} {des}", alpha=alpha)

def plot_3d(traces, labels, states_pair):
    dim = traces.shape[1]
    len = traces.shape[0]
    X0 = traces[labels == 0]
    X1 = traces[labels == 1]



    ax.scatter(X0[:, 0], X0[:, 1], X0[:, 2], color="red", label=f"state:{states_pair[0]}")
    ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], color="blue", label=f"state:{states_pair[1]}", marker='^', alpha=0.005)
def run_lda(states_pair, batch_size, noise):
    labels, traces = gen_am(batch_size, states_pair, q=q, noise=noise)
    clf = lda()
    clf.fit(traces, labels)
    a_labels, a_traces = gen_am(5000, states_pair, q=q, noise=noise)
    score = clf.score(a_traces, a_labels)
    return score

if __name__ == '__main__':
    batch_size = 50000
    dim = 2
    q = 13

    #Boolean mask
    st0 = np.array([255])
    st1 = np.array([0])
    # st0 = np.random.randint(0, 256, (1, ))
    # st1 = np.random.randint(0, 256, (1, ))
    # states_pair = np.vstack((st0, st1))
    # labels, traces = gen_bm8(batch_size, states_pair, noise=0.1,  n_random=0)
    # clf = lda()
    # clf.fit(traces, labels)
    # a_labels, a_traces = gen_bm8(1000, states_pair, n_random=0)
    # print(clf.score(a_traces, a_labels))
    # plot_(traces, labels, states_pair, alpha=1, des="")
    # plt.ylim(-1, 1)
    # plt.show()

    #Arith mask
    # st0 = np.ones((dim, 1))*(-2)
    # st1 = np.zeros((dim, 1))
    # st0 = np.random.randint(-2, 3, (dim, 1))
    # st1 = np.random.randint(-2, 3, (dim, 1))
    # states_pair = np.hstack((st0, st1)).T
    # labels, traces = gen_am(batch_size, states_pair, q=q, noise=0)
    # clf = lda()
    # clf.fit(traces, labels)
    # a_labels, a_traces = gen_am(1000, states_pair, q=q, noise=0)
    # print(clf.score(a_traces, a_labels))
    # plot_(traces, labels, states_pair, alpha=0.1, des="")
    # plt.show()























    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # st0 = np.zeros((dim, 1))
    # st0[np.arange(0, dim, 2)] = -2
    # st1 = np.zeros((dim, 1))
    # st1[np.arange(1, dim, 2)] = 1
    # print(st0, st1)
    # st0 = np.ones((dim, 1))*(-2)
    # st0[-128:] = 0
    # # st1 = np.zeros((dim, 1))
    # st1 = np.ones((dim, 1))
    # states_pair = np.array([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # st0 = np.random.randint(-2, 3, (dim, 1))
    # st1 = np.random.randint(-2, 3, (dim, 1))
    # print(st0, st1)
    # states_pair = np.concatenate((st0, st1), axis=1).astype(np.int16).T
    # print(states_pair.shape)
    # labels, traces = gen_bm8(batch_size, states_pair)
    # labels, traces = gen_am(batch_size, states_pair, q=q, noise=0.1)


    # x0 = traces[labels==0]
    # x1 = traces[labels==1]
    # mx0 = np.mean(x0, axis=0)
    # mx1 = np.mean(x1, axis=0)
    # print(mx0, mx1)
    # print(np.mean(x0),  np.mean(x1), np.mean(x0) - np.mean(x1))

    # ax.scatter(mx0[ 0], mx0[ 1], mx0[ 2], color="brown", label=f"mean:{states_pair[0]}")
    # ax.scatter(mx1[ 0], mx1[ 1], mx1[ 2], color="orange", label=f"mean:{states_pair[1]}")

    # plot_(traces, labels, states_pair, alpha=0.1)
    # plot_3d(traces, labels, states_pair)
    # clf = lda()
    # clf.fit(traces, labels)
    # print(clf.score(traces, labels))
    # clf_ = LDA(2, 1, 100)
    # clf_.fit_u(traces, labels)
    # clf_.solve()
    # a_labels, a_traces = gen_bm8(1, states_pair)


    # print(clf_.predict_proba(a_traces)[:, a_labels[0]])


    # a_labels, a_traces = gen_bm8(1000, states_pair)

    # a_labels, a_traces = gen_am(1000, states_pair, q=q, noise=0.1)
    # print(clf.score(a_traces, a_labels))
    # project_X = clf.transform(a_traces)
    # print(project_X.shape)
    # plot_(project_X, a_labels, states_pair, alpha=1, des="project")
    # plt.ylim(-1, 1)
    # plt.legend()
    # plt.show()
    # batch_size = 50000
    # dim = 256
    # q = 3329
    # stat = {}
    # stat['pair'] = []
    # stat['score'] = []
    # noise = 0.5
    #
    # with trange(5) as t:
    #     for i in t:
    #         st0 = np.ones((dim, 1))*(-2)
    #         st1 = np.zeros((dim, 1))
    #         # st0 = np.random.randint(-2, 3, (dim, 1))
    #         # st1 = np.random.randint(-2, 3, (dim, 1))
    #         t.set_description(f'Process pair {i}')
    #         states_pair = np.concatenate((st0, st1), axis=1).astype(np.int16).T
    #         score = run_lda(states_pair, batch_size, noise=noise)
    #         print(score)
    #         stat['pair'].append(states_pair)
    #         stat['score'].append(score)
    # with open(f"stat_{q}_{noise}_float.pkl", "wb") as f:
    #     pickle.dump(stat, f)
