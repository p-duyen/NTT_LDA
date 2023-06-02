import numpy as np
from matplotlib import pyplot as plt


s_range = [-2, -1, 0, 1, 2]
prior_s = [0.0625, 0.25,   0.375,  0.25,   0.0625]
SIGMA = [0.51672043, 1.63401346, 5.16720427]


def count_1(x):
    return int(x).bit_count()
fcount = np.vectorize(count_1)
def HW(x):
    return fcount(x)

def pdf_normal(x, mu, sigma):
    print(x.shape)
    ep = (x-mu)/sigma
    ep = -ep**2/2
    return np.exp(ep) / (sigma * np.sqrt(2*np.pi))

def ent_s(prior_s):
    """Entropy for prior proba
    """
    log_2p = np.log2(prior_s)
    return -(prior_s*log_2p).sum()
# print(ent_s([1/2, 1/2]))
def gen_states(n_states, n_coeffs):
    states = np.empty((n_states, n_coeffs), dtype=np.int32)
    for i in range(n_states):
        states[i] = np.random.choice(s_range, n_coeffs, p=prior_s)
    return states

def gen_shares(states, n_shares, q, op="sub"):
    shares = {}
    masked_state = np.zeros(states.shape)
    for i in range(n_shares-1):
        tmp = np.random.randint(0, q, size=states.shape, dtype=np.int32)
        masked_state += tmp
        # shares[f"S{i}"] = (q-tmp)%q
        shares[f"S{i}"] = tmp if op=="sub" else (q-tmp)%q
    shares[f"S{n_shares-1}"] = (states - masked_state)%q
    return shares

def gen_leakages(shares, sigma, combif=None):
    Lis = []
    for share, share_val in shares.items():
        Lis.append(HW(share_val) + np.random.normal(0, sigma, size=share_val.shape))
        # print(share_val.shape)
    Lis = np.array(Lis)
    n_shares, n_samples, n_coeffs = Lis.shape
    if combif is None:
        return Lis
    if combif == "prod":
        L = np.ones((n_samples, n_coeffs))
        for i in range(n_shares):
            L *= Lis[i]
        return L
    if combif == "norm_prod":
        L = np.ones((n_samples, n_coeffs))
        for i in range(n_shares):
            L *= Lis[i] - np.mean(Lis[i], axis=0)
        return L
    if combif == "abs_diff":
        L = np.zeros((n_samples, n_coeffs))
        for i in range(n_shares):
            L = Lis[i] - L
        return np.abs(L)
    if combif == "sum":
        L = np.zeros((n_samples, n_coeffs))
        for i in range(n_shares):
            L = Lis[i] + L
        return L
def leakage_fix(L):
    n_shares, n_traces, n_coeffs = L.shape
    L_fix = np.empty((n_traces, n_coeffs*n_shares))
    for i in range(n_shares):
        L_fix[:, n_coeffs*i:(n_coeffs*i+n_coeffs)] = L[i]
    return L_fix
def gen_sigma(idx=0):
    small_logs2 = np.linspace(-3, -1, 3)
    med_logs2 = np.linspace(-0.75, 1, 5)
    large_logs2 = np.linspace(1.25, 2, 4)
    log_sigma_2 = np.hstack((small_logs2, med_logs2, large_logs2))
    sigma_2_10 = np.power(10, log_sigma_2)
    sigma = np.sqrt(sigma_2_10)
    return log_sigma_2[idx:], sigma[idx:]

def gen_sigma_(idx=0):
    small_logs2 = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 0.75, 1, 1.25])
    sigma_2_10 = np.power(10, small_logs2)
    sigma = np.sqrt(sigma_2_10)
    return small_logs2[idx:], sigma[idx:]
# print(gen_sigma_()[1][3:5])

def pdf_l_given_s(l, q, sigma=0.1):
    Zq = np.arange(q)
    hw_set = HW(Zq)
    pdf = pdf_normal(l, hw_set, sigma)
    return pdf/pdf.sum(axis=1, keepdims=True)

def readout_sr():
    sigma_2, sigma = gen_sigma()
    sigma_2 = sigma_2
    sigma = sigma
    combif = "norm_prod"
    n_profiling = 10000
    sr_range = []
    sig_range = []
    for i, sig in enumerate(sigma):
        if sigma_2[i] <= -1:
            n_attacks = np.arange(1, 20, 2)
        elif sigma_2[i] <= 1:
            n_attacks = np.arange(5, 500, 20)
        else:
            n_attacks = np.arange(10, 5000, 200)
        with open(f"sr/{combif}_{n_profiling}_{sig:0.4f}.npy", "rb") as f:
            sr = np.load(f)
        success_point = np.argmax(sr)
        max_sr = np.max(sr)
        if max_sr > 0.9:
            sr_range.append(np.log10(n_attacks[np.argmax(sr)]))
            sig_range.append(i)
        # sr_range.append(np.max(sr))
        print(sig, sr[success_point], n_attacks[success_point])
        if i in [9, 10, 11]:
            n_attacks_ = np.arange(10, 3000, 200)
            n_a = [50000, 100000, 200000]
            with open(f"sr/{combif}_{50000}_{sig:0.4f}.npy", "rb") as f_:
                sr_ = np.load(f_)
                # plt.plot(n_attacks_, sr_, linestyle="dashed", label=f"{sig:0.4f} 50k")
                success_point = np.argmax(sr_)
                print("50000", sig, sr_[success_point], n_attacks[success_point])
            with open(f"sr/{combif}_{100000}_{sig:0.4f}.npy", "rb") as f_:
                sr_ = np.load(f_)
                # plt.plot(n_attacks_, sr_, linestyle="dotted", label=f"{sig:0.4f} 100k")
                success_point = np.argmax(sr_)
                print("100000", sig, sr_[success_point], n_attacks[success_point])
            with open(f"sr/{combif}_{200000}_{sig:0.4f}.npy", "rb") as f_:
                sr_ = np.load(f_)
                # plt.plot(n_attacks_, sr_, linestyle="dotted", label=f"{sig:0.4f} 200k")
                success_point = np.argmax(sr_)
                print("200000", sig, sr_[success_point], n_attacks[success_point])

        # plt.plot(n_attacks, sr, label=f"{sig:0.4f}")
        # plt.title(f"n_profiling: {n_profiling}")
    plt.plot(sigma_2[sig_range], sr_range)
    plt.scatter(sigma_2[sig_range], sr_range)
    plt.legend()
    plt.show()
# print(np.arange(10, 3000, 600).shape)
# readout_sr()
def readout():
    #==================2 shares===============
    sigma_2, sigma = gen_sigma()
    combif = "abs_diff"
    n_profiling = 10000
    sr_range = []
    sig_range = []

    inc_sig = []
    for i, sig in enumerate(sigma):
        if sigma_2[i] <= -1:
            n_attacks = np.arange(1, 20, 2)
        elif sigma_2[i] <= 1:
            n_attacks = np.arange(5, 500, 20)
        else:
            n_attacks = np.arange(10, 5000, 200)
        with open(f"sr/{combif}_{n_profiling}_{sig:0.4f}.npy", "rb") as f:
            sr = np.load(f)
        plt.plot(n_attacks, sr, label=f"{sig:0.4f}")
        # success_point = np.argmax(sr)
        # max_sr = np.max(sr)
        # if max_sr > 0.9:
        #     sr_range.append(np.log10(n_attacks[np.argmax(sr>0.9)]))
        #     sig_range.append(i)

    # plt.plot(sigma_2[sig_range], sr_range, label="2 shares 10k")
    # plt.scatter(sigma_2[sig_range], sr_range)
    # #==================2 shares high noise==========
    # n_a = [50000, 100000, 200000]
    # des = [50, 100, 200]
    # n_attacks_ = np.arange(10, 3000, 200)
    # print(sigma_2[-3:])
    # for i_na, na in enumerate(n_a):
    #     inc_sr = []
    #     inc_sig = []
    #     for i_s, sig in enumerate(sigma[-3:]):
    #         with open(f"sr/{combif}_{na}_{sig:0.4f}.npy", "rb") as f:
    #             sr = np.load(f)
    #         max_sr = np.max(sr)
    #         if max_sr > 0.9:
    #             inc_sr.append(np.log10(n_attacks_[np.argmax(sr>0.9)]))
    #             inc_sig.append(i_s)
    #     # plt.plot(sigma_2[-3:], inc_sr, label=f"2 shares {des[i_na]}k")
    #     print(inc_sr, inc_sig)
    #     plt.scatter(sigma_2[-3:][inc_sig], inc_sr, label=f"2 shares {des[i_na]}k")
    # #==================3 shares===============
    # sigma = [0.0316, 0.1, 0.3162, 0.5623]
    # sigma_2 = [-3, -2, -1]
    # n_profiling = 200000
    # n_attacks = np.arange(10, 3000, 600)
    # combif = "norm_prod"
    # n_shares = 3
    # na3 = []
    # for sig in sigma:
    #     with open(f"sr/{combif}_{n_shares}_{n_profiling}_{sig:0.4f}.npy", "rb") as f:
    #         sr = np.load(f)
    #     na3.append(np.log10(n_attacks[np.argmax(sr>0.88)]))
    #
    # plt.plot(sigma_2, na3, label=f"3 shares 200k")
    # plt.scatter(sigma_2, na3)
    # print(list(plt.xticks()[0]))
    # plt.xticks(list(plt.xticks()[0][1:]) + [1.5, 1.75, 2])
    plt.legend()
    plt.show()
# print(1/0.03162277660168379)
# readout()
def readout_mi(q):
    N_SHARES = [2]
    combi_fs = ["prod", "norm_prod", "abs_diff"]
    color = ["orange", "pink", "green"]
    color_add = ["olive", "deeppink", "lime"]
    # sigma_2, sigmas = gen_sigma_()
    # sigma_2_, sigmas_ = gen_sigma_()
    for n_shares in N_SHARES:
        for i_c, combif in enumerate(combi_fs):
            with open(f"mi/{q}_{n_shares}_{combif}.npy", "rb") as f:
                sigma_2 = np.load(f)
                mi = []
                for sig in sigma_2:
                    mi.append(np.load(f))

            plt.plot(sigma_2, mi, label=f"{combif}_{n_shares}_sub", color=color[i_c])
            plt.scatter(sigma_2, mi, color=color[i_c], s=10)
            with open(f"mi/{q}_{n_shares}_{combif}_add.npy", "rb") as f:
                sigma_2 = np.load(f)
                mi = []
                for sig in sigma_2:
                    mi.append(np.load(f))
            if combif == "abs_diff":
                plt.plot(sigma_2[:4], mi[:4], label=f"{combif}_{n_shares}_add", color=color_add[i_c])
                plt.scatter(sigma_2[:4], mi[:4], color=color[i_c], s=10)
            else:
                plt.plot(sigma_2, mi, label=f"{combif}_{n_shares}_add", color=color_add[i_c])
                plt.scatter(sigma_2, mi, color=color[i_c], s=10)
        plt.legend()
        plt.show()
# print(gen_sigma_()[1])
# readout_mi(23)
# print(np.arange(1000, 5000, 1000))
def matrix_print(A):
    for row in A:
        for col in row:
            print(f"{col:0.4f}", end=" ")
        print("\n")
snr = np.array([0.1, 1, 10])
def sigma_snr():
    snr_L = np.linspace(10, 1, 3)
    snr_S = np.linspace(1, 0.1, 3)
    snr = np.hstack((snr_L, snr_S[1:]))
    sigma = np.sqrt(2.67/snr)
    return snr, sigma
SIGMA = [0.51672043, 1.63401346, 5.16720427]
N_p = [200000]
NP = []
PI = []
def pi_3(sigma):
    with open(f"/home/tpay/Desktop/WS/NTT_DIST/pi/pi_3329_3_{sigma}_norm_prod.npy", "rb") as f:
        Np = np.load(f)
        print(Np)
        for n_p in Np:
            NP.append(n_p)
            pi = np.load(f)
            # pi = pi[pi>0]
            print(n_p, pi.mean())
            PI.append(pi.mean())
    with open(f"/home/tpay/Desktop/WS/NTT_DIST/pi/pi_3329_3_{sigma}_norm_prod_100000.npy", "rb") as f:
        Np = np.load(f)
        # print(Np)
        for n_p in Np:
            NP.append(n_p)
            pi = np.load(f)
            # pi = pi[pi>0]
            PI.append(pi.mean())
    #         # print(pi)
    #         # print(n_p, pi,  pi.mean())
    print(NP)
    print(PI)
    plt.plot(NP, PI)
    plt.scatter(NP, PI)
    plt.show()
pi_3(SIGMA[2])
# Na_1 = np.arange(1, 30, 2)
# Na_2 = np.arange(1, 50, 3)
# Na_3 = np.arange(100, 1000, 50)
# print(Na_3, Na_3.shape)
