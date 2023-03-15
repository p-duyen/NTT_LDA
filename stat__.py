import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import norm
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.interpolate import make_interp_spline
from gen_data import *

def count_1(x):
    return int(x).bit_count()

def HW(x):
    fcount = np.vectorize(count_1)
    return fcount(x)
def LSB(x):
    return x%2
def bm8_gen(n_samples, s, noise=0, comf=None):
    r = np.random.randint(0, 256, (n_samples, ))
    ms = s^r
    l0 = HW(r) + np.random.normal(0, noise, (n_samples, ))
    l1 = HW(ms) + np.random.normal(0, noise, (n_samples, ))
    if comf=="prod":
        l = (l0-np.mean(l0))*(l1 -np.mean(l1))
    elif comf=="abs_diff":
        l = np.absolute(l0-l1)
    elif comf=="sum":
        l = l0+l1
    elif comf==None:
        l = np.column_stack((l0, l1))
    y = np.column_stack((r, ms))
    return y, l
def am_gen(n_samples, s, q, noise=0, comf=None):
    r = np.random.randint(0, q, (n_samples))
    t = q - r
    ms = (s-r)%q
    l0 = HW(r) + np.random.normal(0, noise, (n_samples, ))
    l1 = HW(ms) + np.random.normal(0, noise, (n_samples, ))
    hwr = HW(r)
    hwms = HW(ms)
    if comf=="prod":
        # l = (l0-np.mean(l0))*(l1 -np.mean(l1))
        l = l0*l1
    elif comf=="norm_prod":
        l = (l0-np.mean(l0))*(l1 -np.mean(l1))
    elif comf=="abs_diff":
        l = np.absolute(l0-l1)
    elif comf=="sum":
        l = l0+l1
    elif comf==None:
        l = np.column_stack((l0, l1))
    y = np.column_stack((r, ms))
    return y, l


def am_gen_(n_samples, s, q, l_model="HW", op="add" ,noise=0, comf=None):
    r = np.random.randint(0, q, (n_samples))
    t = (q - r)%q
    ms = (s - r)%q
    print(l_model)
    if op == "add":
        if l_model == "HW":
            l0 = HW(r) + np.random.normal(0, noise, (n_samples, ))
        elif l_model == "LSB":
            l0 = LSB(r) + np.random.normal(0, noise, (n_samples, ))
        else:
            l0 = r + np.random.normal(0, noise, (n_samples, ))
    else:
        print("sub")
        if l_model == "HW":
            l0 = HW(t) + np.random.normal(0, noise, (n_samples, ))
        elif l_model == "LSB":
            l0 = LSB(t) + np.random.normal(0, noise, (n_samples, ))
        else:
            l0 = t + np.random.normal(0, noise, (n_samples, ))
    if l_model == "HW":
        l1 = HW(ms) + np.random.normal(0, noise, (n_samples, ))
    elif l_model == "LSB":
        l1 = LSB(ms) + np.random.normal(0, noise, (n_samples, ))
    else:
        l1 = ms + np.random.normal(0, noise, (n_samples, ))
    if comf=="prod":
        l = l0*l1
    elif comf=="norm_prod":
        l = (l0-np.mean(l0))*(l1 -np.mean(l1))
    elif comf=="abs_diff":
        l = np.absolute(l0-l1)
    elif comf=="sum":
        l = l0+l1
    elif comf==None:
        l = np.column_stack((l0, l1))
    y = np.column_stack((r, ms))
    return y, l
def hist2d_norm():
    xedges = [0,1,3,5]
    yedges = [0,2,3,4,6]
    # create edges of bins

    x = np.random.normal(2, 1, 100) # create random data points
    y = np.random.normal(1, 1, 100)
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges), normed=True)
    areas = np.matmul(np.array([np.diff(xedges)]).T, np.array([np.diff(yedges)]))
    # setting normed=True in histogram2d doesn't seem to do what I need

    fig = plt.figure(figsize=(7, 3))
    im = plt.imshow(H*areas, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar(im)
    plt.show()

def dist_1d(comf):
    n_samples = 1000000
    noise = 30
    # s_range = [0, 1, 3, 7, 15]
    s_range = [-2, -1, 0, 1, 2]
    q = 3329
    # comf="prod"
    fig, axs = plt.subplots(1, 5, sharey=True)
    fig.subplots_adjust(top=0.5)
    # fig.suptitle(f"Boolean_Mask noise: {noise}, combi_f = {comf}", fontsize=12, y=0.55)
    fig.suptitle(f"q = {q}, noise = {noise}, combi_f = {comf}", fontsize=12, y=0.55)
    mean_prod = []
    mean_nprod = []

    for (s, ax) in zip(s_range, axs):
        st = np.array([s])
        y, l = am_gen(n_samples, st, q=q, noise=noise, comf=comf)
        hist = np.histogram(l, bins=500, density=True)
        X = hist[1][:-1]

        Y = hist[0]
        X_Y_Spline = make_interp_spline(X, Y)
        X_ = np.linspace(X.min(), X.max(), 1000)
        Y_ = X_Y_Spline(X_)

        ax.plot(X, Y)
        ticks = [int(X[0]), np.mean(l),  int(X[-1])]
        ax.set_xticks(ticks)
        ax.get_xticklabels()[1].set_color("red")
        # print(f"prod s: {s}, l: {l}, {np.mean(l)}")
        mean_prod.append(np.mean(l))
        ax.set(adjustable='box')
        ax.set_title(f"s={st[0]}")
    plt.show()


def histo_2D():
    n_samples = 500000
    noise = 0.2
    # s_range = [0, 1, 3, 7, 5]
    s_range = [-2, -1, 0, 1, 2]
    q = 3329

    fig, axs = plt.subplots(1, len(s_range), sharey=True)
    # fig.suptitle(f"q = {q}, noise = {noise}, combi_f = {comf}", fontsize=12)
    fig.suptitle(f"q = {q}", fontsize=12)
    # fig.suptitle(f"Boolean_Mask noise: 0.5", fontsize=12)
    fig.subplots_adjust(top=1.5)


    for (s, ax) in zip(s_range, axs):
        st = np.array([s])
        y, l = am_gen_(n_samples, st, q=q, noise=noise, comf=None)
        # y, l = bm8_gen(n_samples, s, noise=noise, comf=None)

        l0 = l[:, 0]
        l1 = l[:, 1]
        ax.hist2d(l0, l1, bins=100, density=True)
        ax.set(aspect='equal', adjustable='box')
        elem_s = (st[0]+q)%q
        ax.set_xlabel("s0")
        ax.set_ylabel("s1")
        # print(f"s: {st}, elem_s: {elem_s}")
        # elem_s = st[0]
        ax.set_title(f"s={st[0]}_{elem_s}_{bin(elem_s)[2:]}_{HW(elem_s)}")
    plt.show()
def histo_2D_():
    n_samples = 500000
    noise = 0.2
    s_range = [-2, -1, 0, 1, 2]
    q = 3329
    ops = ["add", "sub"]
    ops_des = ["s1 = (s-s0)%q", "s1 = (s+s0')%q"]
    x_a = ["s0", "s0'"]

    fig, axs = plt.subplots(1, len(s_range), sharey=True)
    fig.suptitle(f"q = {q}={q:012b}", fontsize=12)
    fig.subplots_adjust(top=1.5)


    for (s, ax) in zip(s_range, axs):
        st = np.array([s])
        y, l = am_gen_(n_samples, st, q=q, noise=noise, comf=None)

        l0 = l[:, 0]
        l1 = l[:, 1]
        ax.hist2d(l0, l1, bins=100, density=True)
        ax.set(aspect='equal', adjustable='box')
        elem_s = (st[0]+q)%q
        ax.set_xlabel("s0")
        ax.set_ylabel("s1")
        # print(f"s: {st}, elem_s: {elem_s}")
        # elem_s = st[0]
        ax.set_title(f"s={st[0]}_{elem_s}_{bin(elem_s)[2:]}_{HW(elem_s)}")
    plt.show()



def histo_2D_op(s_range, q, l_model):
    n_samples = 500000
    noise = 0.2
    ops = ["add", "sub"]
    ops_des = ["s1 = (s-s0)%q", "s1 = (s+s0')%q"]
    x_a = ["s0", "s0'"]
    fs_subfig = 8
    fs_label = 8
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"q = {q}_{q:03b} leakage: {l_model}")

    subfigs = fig.subfigures(nrows=len(ops), ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f"{ops_des[row]}", y=0.98)
        axs = subfig.subplots(nrows=1, ncols=s_range.shape[0], sharey=True)

    # fig.suptitle(f"q = {q}, noise = {noise}, combi_f = {comf}", fontsize=12)
        for (s, ax) in zip(s_range, axs):
            st = np.array([s])
            y, l = am_gen_(n_samples, st, l_model=l_model, op=ops[row], q=q, noise=noise, comf=None)
            # y, l = bm8_gen(n_samples, s, noise=noise, comf=None)
            l0 = l[:, 0]
            l1 = l[:, 1]
            # print(l0)
            ax.set_xlabel(f"{x_a[row]}", fontsize=fs_subfig)
            ax.set_ylabel("s1", fontsize=fs_subfig)
            ax.hist2d(l0, l1, bins=100, density=True)
            # ticks = np.arange(q)
            # ax.set_xticks(ticks)
            ax.set(aspect='equal', adjustable='box')
            elem_s = (st[0]+q)%q

            # print(f"s: {st}, elem_s: {elem_s}")
            # elem_s = st[0]
            ax.set_title(f"s={st[0]}={elem_s:04b}_{HW(st)}", fontsize=fs_label)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.subplots_adjust(pad=-5.0)
    plt.show()
def histo_2D_op_tightlayout(s_range, q, l_model, n_cols, op=0):
    n_samples = 500000
    noise = 0.2

    ops = ["add", "sub"]
    ops_des = ["s1 = (s-s0)%q", "s1 = (s+s0')%q"]
    x_a = ["s0", "s0'"]
    fs_subfig = 8
    fs_label = 8
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"q = {q}={q:08b} leakage: {l_model} {ops_des[op]}", fontsize=12)

    last_row = 0 if q%n_cols==0 else 1
    nrows = q//n_cols + last_row
    print(nrows, last_row)

    subfigs = fig.subfigures(nrows=nrows, ncols=1)
    for row, subfig in enumerate(subfigs):
        # subfig.suptitle(f"{ops[row]}, {ops_des[row]}")
        print(f"row: {row}")
        if row < nrows - 1:
            axs = subfig.subplots(nrows=1, ncols=n_cols, sharey=True)
            for (s, ax) in zip(s_range[row*n_cols:(row+1)*n_cols ], axs):
                st = np.array([s])
                y, l = am_gen_(n_samples, st, l_model=l_model, op=ops[op], q=q, noise=noise, comf=None)
                l0 = l[:, 0]
                l1 = l[:, 1]
                ax.set_xlabel(f"{x_a[op]}", fontsize=fs_subfig)
                ax.set_ylabel("s1", fontsize=fs_subfig)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.hist2d(l0, l1, bins=100, density=True)
                ax.set(aspect='equal', adjustable='box')
                elem_s = (st[0]+q)%q
                ax.set_title(f"s={st[0]}={elem_s:012b}_HW={HW(st)}", fontsize=fs_label)
        else:
            axs = subfig.subplots(nrows=1, ncols=n_cols if q%n_cols == 0 else q%n_cols, sharey=True)
            for (s, ax) in zip(s_range[row*n_cols:], axs):
                st = np.array([s])
                y, l = am_gen_(n_samples, st, l_model=l_model, op=ops[op], q=q, noise=noise, comf=None)
                l0 = l[:, 0]
                l1 = l[:, 1]
                ax.set_xlabel(f"{x_a[op]}", fontsize=fs_subfig)
                ax.set_ylabel("s1", fontsize=fs_subfig)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.hist2d(l0, l1, bins=100, density=True)
                ax.set(aspect='equal', adjustable='box')
                elem_s = (st[0]+q)%q
                ax.set_title(f"s={st[0]}={elem_s:012b}_HW={HW(st)}", fontsize=fs_label)
    plt.show()

def hist2d_Q(s_range, q, l_model, n_cols, op=0):
    n_samples = 500000
    noise = 0.2

    ops = ["add", "sub"]
    ops_des = ["s1 = (s-s0)%q", "s1 = (s+s0')%q"]
    x_a = ["s0", "s0'"]
    fs_subfig = 8
    fs_label = 8

    last_row = 0 if q%n_cols==0 else 1
    nrows = q//n_cols + last_row
    sub_fig_rows = 3
    sub_figs = nrows//4 + 1
    print(f"NUMBER of fig: {sub_figs} row in each fig: {sub_fig_rows}")
    for i in range(sub_figs):
        print(f"Fig: {i}")
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"q = {q}={q:08b} leakage: {l_model} {ops_des[op]}", fontsize=12)
        subfigs = fig.subfigures(nrows=sub_fig_rows, ncols=1)
        for sub_row, subfig in enumerate(subfigs):
            row = i*sub_fig_rows + sub_row
            print(f"row: {row}")
            if row < nrows - 1:
                axs = subfig.subplots(nrows=1, ncols=n_cols, sharey=True)
                for (s, ax) in zip(s_range[row*n_cols:(row+1)*n_cols ], axs):
                    st = np.array([s])
                    y, l = am_gen_(n_samples, st, l_model=l_model, op=ops[op], q=q, noise=noise, comf=None)
                    l0 = l[:, 0]
                    l1 = l[:, 1]
                    ax.set_xlabel(f"{x_a[op]}", fontsize=fs_subfig)
                    ax.set_ylabel("s1", fontsize=fs_subfig)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    ax.hist2d(l0, l1, bins=100, density=True)
                    ax.set(aspect='equal', adjustable='box')
                    elem_s = (st[0]+q)%q
                    ax.set_title(f"s={st[0]}={elem_s:012b}_HW={HW(st)}", fontsize=fs_label)
            else:
                axs = subfig.subplots(nrows=1, ncols=n_cols if q%n_cols == 0 else q%n_cols, sharey=True)
                for (s, ax) in zip(s_range[row*n_cols:], axs):
                    st = np.array([s])
                    y, l = am_gen_(n_samples, st, l_model=l_model, op=ops[op], q=q, noise=noise, comf=None)
                    l0 = l[:, 0]
                    l1 = l[:, 1]
                    ax.set_xlabel(f"{x_a[op]}", fontsize=fs_subfig)
                    ax.set_ylabel("s1", fontsize=fs_subfig)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    ax.hist2d(l0, l1, bins=100, density=True)
                    ax.set(aspect='equal', adjustable='box')
                    elem_s = (st[0]+q)%q
                    ax.set_title(f"s={st[0]}={elem_s:012b}_HW={HW(st)}", fontsize=fs_label)
        plt.savefig(f"dist/HW_3329_add_{i}.png")



def histo_2d_overlap():
    n_samples = 500000
    noise = 0.2
    # s_range = [0, 1, 3, 7, 5]
    s_range = [-1, 0]
    colors = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    q = 3329
    comf="norm_prod"


    #jointly distrib
    #
    # fig, ax = plt.subplots()
    # fig.suptitle(f"q = {q}, noise = {noise}, combi_f = {comf}", fontsize=12)
    # fig.suptitle(f"Boolean_Mask noise: 0.5", fontsize=12)
    # fig.subplots_adjust(top=1.5)


    for (s, cm) in zip(s_range, colors):
        st = np.array([s])
        print(st, mpl.colormaps[cm])
        y, l = am_gen(n_samples, st, q=q, noise=noise, comf=None)
        # y, l = bm8_gen(n_samples, s, noise=noise, comf=None)

        l0 = l[:, 0]
        l1 = l[:, 1]
        plt.hist2d(l0, l1, bins=100, density=True, cmap=mpl.colormaps[cm], label=f"s:{st[0]}", alpha=0.5)
        # plt.set(aspect='equal', adjustable='box')
        elem_s = (st[0]+q)
        # print(f"s: {st}, elem_s: {elem_s}")
        # elem_s = st[0]
        # ax.set_title(f"s={st[0]}_{HW(elem_s)}")
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    # comf=["prod", "norm_prod", "abs_diff"]
    # [dist_1d(f) for f in comf]
    q = 3329

    s_range = np.arange(q, dtype=np.int16)
    l_model = "HW"
    # s_range = np.array([-2, -1, 0, 1, 2])
    # print(f"                 q = {q}              ")
    # print(LSB(s_range))
    # s0_range = np.arange(q//2)
    # for s in s_range:
    #     print(f"s = {s}")
    #     for s0 in s0_range:
    #         s0_ = q - s0
    #         s1 = (s-s0)%q
    #         print(f"       s_1={s1:02d}={s1:06b}~{HW([s1])}, s_0={s0}={s0:06b}~{HW([s0])}, s_0'={s0_:02d}={s0_:06b}~{HW([s0_])}")

    # histo_2D_op(s_range, q, l_model)
    # histo_2D_op_tightlayout(s_range, q, l_model, n_cols=8, op=0)
    hist2d_Q(s_range, q, l_model, n_cols=6, op=1)
    # histo_2D()
    # histo_2d_overlap()
