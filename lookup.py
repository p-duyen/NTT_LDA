import numpy as np
import pickle
from LDA_DIST_small import *


def ass():
    qs = [7, 13, 19]
    n_profilings = [500, 1000, 5000, 10000, 50000]
    for q in qs:
        for n in n_profilings:
            with open(f"sum_{q}_{n}.pkl", "rb") as f:
                sum = pkl.load(f)
                states_set = np.array(sum['states'])
                scores_set = np.array(sum['scores'])
                print(f"{q}, {n}, {np.average(scores_set)}")
                masks = np.array([states_set[i][0]== states_set[i][1] for i in range(len(states_set))])
                iden_pair =  masks[:, 0]& masks[:, 1]
                print(f"{q}, {n}, {np.average(scores_set[~iden_pair])}")
                # # print(np.argsort(scores_set) )
                # max_pairs = scores_set.argsort()[-2:]
                # print(scores_set[max_pairs])
                # print(states_set[max_pairs])
                # print(np.sort(states_set[max_pairs]))
                print('++++++++++++++++')
                # print(np.sort(scores_set))

def cluster_plot():
    q = 7
    N_SAMPLES = 5000
    with open("states_2.npy", "rb") as f:
        states_set = np.load(f)
    data_dump = {}
    data_dump['states'] = []
    data_dump['traces'] = []
    data_dump['labels'] = []
    data_dump['scores'] = []
    c0 = []
    c1 = []
    with trange(states_set.shape[0]) as t:
        for i in t:
            states_pair = states_set[i]
            p_data = gen_data_m(states_pair, n_random=0,  n_samples=N_SAMPLES, q=q)
            X = p_data['traces']
            y = p_data['labels']
            data_dump['states'].append(states_pair)
            data_dump['traces'].append(X)
            data_dump['labels'].append(y)
            clf = profiling(p_data)
            projection = clf.transform(X)
            a_data = gen_data_m(states_pair, n_samples=1000, q=q)
            score = attack(a_data, clf)
            data_dump['scores'].append(score)
            c0.append(projection[y == 0])
            c1.append(projection[y == 1])

    with open(f"data_dump_{q}_{N_SAMPLES}_2.pkl", "wb") as fd:
        pickle.dump(data_dump, fd)
    c_0 = np.array(c0).reshape((c0.shape[0]*c0.shape[1], ))
    c_1 = np.array(c1).reshape((c0.shape[0]*c0.shape[1], ))
    y0 = np.ones(c_0.shape[0])*0.1
    y1 = np.ones(c_0.shape[0])*-0.1
    plt.plot(c_0, y0, color='red', label=0)
    plt.plot(c_1, y1, color='blue', label=0)
    plt.ylim(-1, 1)
    plt.legend()
    plt.show()
if __name__ == '__main__':
    cluster_plot()
