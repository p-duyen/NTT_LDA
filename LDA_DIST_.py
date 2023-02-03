import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse
from scalib.modeling import LDAClassifier
from tqdm import tqdm
import pickle

def HW(x):
    fbin = np.vectorize(np.binary_repr)
    bin_x = [fbin(xi) for xi in x]
    fcount = np.vectorize(np.char.count)
    hw = np.array([fcount(xi, '1') for xi in bin_x])
    return hw

def gen_data(s_num, states, profiling=True):
    states = np.concatenate((np.random.randint(-2, 2, (1, 256), dtype=np.int16), np.random.randint(-2, 2, (1, 256 ), dtype=np.int16)), axis=0)
    data = {}
    data['states']= states
    if profiling:
        print(f"===========GEN DATA for profiling {s_num} with batchsize :{batch_size}============")
        batch_num = s_num//batch_size
        for i in tqdm(range(0, batch_num), desc ="Generating"):
            state_selects = np.concatenate((np.ones((batch_size + batch_size%2)//2, dtype=np.int16), np.zeros((batch_size-(batch_size + batch_size%2)//2), dtype=np.int16)), axis=0)
            np.random.shuffle(state_selects)
            st = states[state_selects]

            data['labels'] = state_selects
            x_0 = np.random.randint(0, q, (batch_size, 256), dtype=np.int16)
            x_1 = (st - x_0)%q

            L0 = HW(x_0) + np.random.normal(0, 0.1, (batch_size, 256))
            L1 = HW(x_1) + np.random.normal(0, 0.1, (batch_size, 256))
            # L = np.concatenate((L0, L1), axis=1)
            L = L0*L1
            data['traces'] = L
            with open(f"p_data_{i}.pkl","wb") as f:
                pickle.dump(data, f)
        return states
    else:
        print(f"===========GEN DATA for attacking {s_num}============")
        state_selects = np.random.randint(0, 2, (s_num, ))
        st = states[state_selects]
        data['labels'] = state_selects
        x_0 = np.random.randint(0, q, (s_num, 256), dtype=np.int16)
        x_1 = (st - x_0)%q

        L0 = HW(x_0) + np.random.normal(0, 0.1, (s_num, 256))
        L1 = HW(x_1) + np.random.normal(0, 0.1, (s_num, 256))
        # L = np.concatenate((L0, L1), axis=1)
        L = L0*L1
        data['traces'] = L
        return data


def profiling(s_num):
    batch_num = s_num//batch_size
    clf = LinearDiscriminantAnalysis()
    for i in tqdm(range(0, batch_num), desc ="Fitting"):
        with open(f"p_data_{i}.pkl","rb") as f:
            data = pickle.load(f)
            labels = data['labels']
            traces = data['traces']
            clf.fit(traces, labels)
    return clf

def attack(s_num, states, clf):
    a_data = gen_data(s_num, states, profiling=False)
    labels = a_data['labels']
    traces = a_data['traces']
    print(f"===========Attack for {s_num}============")
    predicts = clf.predict(traces)
    matches = predicts==labels
    print(f"Correct guesses: {matches.sum()}, accuracy: {clf.score(traces, labels)}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-profiling', type=int, required=True)
    parser.add_argument('-attack', type=int, required=True)
    args = parser.parse_args()
    p_size = args.profiling
    a_size = args.attack
    q = 3329
    batch_size = 50000

    # print(np.concatenate((np.ones((batch_size + batch_size%2)//2, dtype=np.int16), np.zeros((batch_size-(batch_size + batch_size%2)//2), dtype=np.int16)), axis=0))

    # states = gen_data(p_size, profiling=True)
    clf = profiling(p_size)
    with open("p_data_0.pkl", "rb") as f:
        data = pickle.load(f)
        states = data['states']
    attack(a_size, states, clf)
