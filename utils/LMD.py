#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin
import numpy as np
from scipy.interpolate import interp1d
from math import ceil, floor
import os
import shutil


# In[2]:


class Hermite(object):
    '''
    分段Hermite插值.
    '''

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.__D = []
        self.derivatives = self.__get_derivatives()

    def __derivative_start(self, x0, x1, x2, y0, y1, y2):
        m1 = (y1 - y0) / (x1 - x0)
        m2 = (y2 - y1) / (x2 - x1)
        m3 = ((x1 - x0) + (x2 - x1))
        D0 = ((2 * (x1 - x0) + (x2 - x1)) * m1 - (x1 - x0) * m2) / m3
        if D0 * m1 >= 0:
            D0 = 0
        elif m2 * m1 >= 0 and abs(D0) > abs(3 * m1):
            D0 = 3 * m1
        return D0

    def __derivative_middle(self, xj_1, xj, xjp1, yj_1, yj, yjp1):
        D1 = (yjp1 - yj) / (xjp1 - xj)
        D2 = (yj - yj_1) / (xj - xj_1)
        if D1 * D2 <= 0:
            Dj = 0
        else:
            w1 = 2 * (xj - xj_1) + (xjp1 - xj)
            w2 = (xj - xj_1) + 2 * (xjp1 - xj)
            Dj = (w1 + w2) / ((w1 / D2) + (w2 / D1))
        return Dj

    def __derivative_end(self, xn_2, xn_1, xn, yn_2, yn_1, yn):
        m1 = (yn - yn_1) / (xn - xn_1)
        m2 = (yn_1 - yn_2) / (xn_1 - xn_2)
        m3 = (xn - xn_1) + (xn_1 - xn_2)
        Dn = ((2 * (xn - xn_1) + (xn_1 - xn_2)) * m1 - (xn - xn_1) * m2) / m3
        if Dn * m1 >= 0:
            Dn = 0
        elif m2 * m1 >= 0 and abs(Dn) > abs(3 * m1):
            Dn = 3 * m1
        return Dn

    def __get_derivatives(self):
        self.__D.append(self.__derivative_start(self.x[0], self.x[1], self.x[2], self.y[0], self.y[1], self.y[2]))

        for j in range(1, len(self.x) - 1):
            self.__D.append(self.__derivative_middle(self.x[j - 1], self.x[j], self.x[j + 1], self.y[j - 1], self.y[j],
                                                     self.y[j + 1]))

        self.__D.append(self.__derivative_end(self.x[-3], self.x[-2], self.x[-1], self.y[-3], self.y[-2], self.y[-1]))
        return self.__D

    def __get_singel_hermite(self, xj_1, xj, yj_1, yj, T_singel, mid):
        Dj_1, Dj = self.__D[mid], self.__D[mid + 1]
        h1 = (1 + 2 * (T_singel - xj_1) / (xj - xj_1)) * yj_1 * ((T_singel - xj) / (xj_1 - xj)) ** 2
        h2 = (1 + 2 * (T_singel - xj) / (xj_1 - xj)) * yj * ((T_singel - xj_1) / (xj - xj_1)) ** 2
        h3 = (T_singel - xj_1) * Dj_1 * ((T_singel - xj) / (xj_1 - xj)) ** 2
        h4 = (T_singel - xj) * Dj * ((T_singel - xj_1) / (xj - xj_1)) ** 2
        H_singel = h1 + h2 + h3 + h4
        return H_singel

    def __Find(self, x, key):  # 找到相应的分段函数的索引
        low = 0
        high = len(x) - 1
        while True:
            mid = int((low + high) / 2)
            if x[mid] <= key < x[mid + 1]:
                break
            elif key >= x[mid + 1]:
                low = mid
            elif key < x[mid]:
                high = mid
        return mid

    def get_all_hermite_values(self, T):
        if T[0] < self.x[0] or T[-1] > self.x[-1]:
            raise IndexError('插值超出范围！')
        H = []
        for i in range(len(T) - 1):
            mid = self.__Find(self.x, T[i])
            H.append(self.__get_singel_hermite(self.x[mid], self.x[mid + 1], self.y[mid], self.y[mid + 1], T[i], mid))
        if T[-1] == self.x[-1]:
            H.append(self.y[-1])
        else:
            mid = self.__Find(self.x, T[-1])
            H.append(self.__get_singel_hermite(self.x[mid], self.x[mid + 1], self.y[mid], self.y[mid + 1], T[i], mid))
        return np.array(H)


# In[3]:


def direct_extension1(xmax, ymax, xmin, ymin, x, y):
    '''
    端点直接延拓方法1，只取前两个极值做平均.
    '''
    xmax.insert(0, x[0] - 10 ** (-10))
    if y[0] > ymax[0]:
        ymax.insert(0, y[0])
    else:
        ymax.insert(0, (ymax[0] + ymax[1]) / 2)

    xmax.append(x[-1] + 10 ** (-10))
    if y[-1] > ymax[-1]:
        ymax.append(y[-1])
    else:
        ymax.append((ymax[-2] + ymax[-1]) / 2)

    xmin.insert(0, x[0] - 10 ** (-10))
    if y[0] < ymin[0]:
        ymin.insert(0, y[0])
    else:
        ymin.insert(0, (ymin[0] + ymin[1]) / 2)

    xmin.append(x[-1] + 10 ** (-10))
    if y[-1] < ymin[-1]:
        ymin.append(y[-1])
    else:
        ymin.append((ymin[-2] + ymin[-1]) / 2)

    return xmax, ymax, xmin, ymin


def direct_extension2(xmax, ymax, xmin, ymin, x, y):
    '''
    端点直接延拓方法2，取一半的极值做平均.
    '''
    xmax.insert(0, x[0] - 10 ** (-10))

    if y[0] > ymax[0]:
        ymax.insert(0, y[0])
    elif len(ymax) % 2 == 0:
        ymax.insert(0, sum(ymax[0:int(len(ymax) / 2)]) / len(ymax[0: \
                                                                  int(len(ymax) / 2)]))
    elif len(ymax) % 2 != 0:
        ymax.insert(0, sum(ymax[0:int((len(ymax) + 1) / 2)]) / len(ymax[0: \
                                                                        int((len(ymax) + 1) / 2)]))

    xmax.append(x[-1] + 10 ** (-10))

    if y[-1] > ymax[-1]:
        ymax.append(y[-1])
    elif len(ymax) % 2 == 0:
        ymax.append((sum(ymax[int(len(ymax) / 2):-1]) + ymax[-1]) / (len(ymax[ \
                                                                         int(len(ymax) / 2):-1]) + 1))
    elif len(ymax) % 2 != 0:
        ymax.append((sum(ymax[int((len(ymax) - 1) / 2):-1]) + ymax[-1]) / (len(ymax[ \
                                                                               int((len(ymax) - 1) / 2):-1]) + 1))

    xmin.insert(0, x[0] - 10 ** (-10))

    if y[0] < ymin[0]:
        ymin.insert(0, y[0])
    elif len(ymin) % 2 == 0:
        ymin.insert(0, sum(ymin[0:int(len(ymin) / 2)]) / len(ymin[0: \
                                                                  int(len(ymin) / 2)]))
    elif len(ymin) % 2 != 0:
        ymin.insert(0, sum(ymin[0:int((len(ymin) + 1) / 2)]) / len(ymin[0: \
                                                                        int((len(ymin) + 1) / 2)]))

    xmin.append(x[-1] + 10 ** (-10))

    if y[-1] < ymin[-1]:
        ymin.append(y[-1])
    elif len(ymin) % 2 == 0:
        ymin.append((sum(ymin[int(len(ymin) / 2):-1]) + ymin[-1]) / (len(ymin[ \
                                                                         int(len(ymin) / 2):-1]) + 1))
    elif len(ymin) % 2 != 0:
        ymin.append((sum(ymin[int((len(ymin) - 1) / 2):-1]) + ymin[-1]) / (len(ymin[ \
                                                                               int((len(ymin) - 1) / 2):-1]) + 1))
    return xmax, ymax, xmin, ymin


# In[4]:


def extremum_coordinates_with_direct_extension(x, y):
    '''
    将极大值，极小值延拓至与数据平齐.
    '''
    ymaxindex, yminindex = np.argmax(np.abs(y)), np.argmin(np.abs(y))
    ymax, ymin, xmax, xmin = [], [], [], []
    for i in [ymaxindex]:
        ymax.append(y[i])
        xmax.append(x[i])
    for i in [yminindex]:
        ymin.append(y[i])
        xmin.append(x[i])
    xmax, ymax, xmin, ymin = direct_extension2(xmax, ymax, xmin, ymin, x, y)
    return xmax, ymax, xmin, ymin


# In[5]:


def e_index(fup, fdown, e):
    '''
    上下包络相交的改正，找出fup,fdown中非常接近的数据<索引>,e是预先设定的一个小量，用于判断fup,fdown的接近程度.
    '''
    eindex = []
    for i in range(1, len(fup) - 1):  # 不检查首尾的点，因为检查出来也无法改正
        if fup[i] - fdown[i] < e:
            eindex.append(i)
    if len(eindex) != 0:
        return eindex, True, len(eindex)
    else:
        return eindex, False, 0


def split_i_index(eindexi):
    '''
    将e_index函数得到的<索引>变成由每个小的、连续的部分组成的列表 (要分成连续的一部分),必须与split_e_index函数配合使用.
    '''
    split_indexi = []
    split_indexi.append(eindexi[0])
    for i in range(1, len(eindexi)):
        if eindexi[i] - eindexi[0] == i:
            split_indexi.append(eindexi[i])
    for i in split_indexi:
        eindexi.remove(i)
    return split_indexi, eindexi


def split_e_index(eindex):
    '''
    将e_index函数得到的<索引>变成由每个小的、连续的部分组成的列表 (要分成连续的一部分),必须与split_i_index函数配合使用.
    '''
    split_indexall = []
    while len(eindex) != 0:
        split_indexi, eindex = split_i_index(eindex)
        split_indexall.append(split_indexi)
    return split_indexall


# ========================================

#
def disjoin(fup, fdown, x, e):
    '''
    上下包络相交的改正.
    '''
    delta_head = fup[0] - fdown[0]  # 这里单独处理一下首尾两个点，防止相交后无法处理
    delta_tail = fup[-1] - fdown[-1]
    if delta_head < e:
        fup[0] = fup[0] + (e - delta_head)
        fdown[0] = fdown[0] - (e - delta_head)
    if delta_tail < e:
        fup[-1] = fup[-1] + (e - delta_tail)
        fdown[-1] = fdown[-1] - (e - delta_tail)
    eindex = e_index(fup, fdown, e)
    count = 0
    eindex_temporary = []
    while eindex[1]:
        sindex = split_e_index(eindex[0])
        # for fup
        for i in sindex:
            p_up = []
            p_upindex = []
            p_up.append(fup[i[0] - 1])
            p_up.append(fup[i[-1] + 1])
            p_upindex.append(x[i[0] - 1])
            p_upindex.append(x[i[-1] + 1])
            p_up = np.array(p_up)
            p_upindex = np.array(p_upindex)
            f1 = interp1d(p_upindex, p_up, kind='linear')(x[(i[0] - 1):(i[-1] + 2)])
            fup[(i[0] - 1):(i[-1] + 2)] = (fup[(i[0] - 1):(i[-1] + 2)] + f1) / 2

        # for fdown
        for i in sindex:
            p_down = []
            p_downindex = []
            p_down.append(fdown[i[0] - 1])
            p_down.append(fdown[i[-1] + 1])
            p_downindex.append(x[i[0] - 1])
            p_downindex.append(x[i[-1] + 1])
            p_down = np.array(p_down)
            p_downindex = np.array(p_downindex)
            f2 = interp1d(p_downindex, p_down, kind='linear')(x[(i[0] - 1):(i[-1] + 2)])
            fdown[(i[0] - 1):(i[-1] + 2)] = (fdown[(i[0] - 1):(i[-1] + 2)] + f2) / 2

        eindex = e_index(fup, fdown, e)
        count += 1
        if count % 5 == 0:
            print('改正相交迭代的次数：', count)
        if count == 50:
            eindex = e_index(fup, fdown, e - 10 ** (-4))
        if count == 200:
            raise ValueError('无法改正上下包络的相交！')
    return fup, fdown


# In[6]:


def ask(t, ui):
    '''
    大循环询问项
    '''
    plt.figure(figsize=(12, 4))
    plt.plot(t, ui)
    plt.xlabel('t')
    plt.ylabel('ui')
    plt.show()
    while True:
        f = input('Stop it?(y/n)')
        if f not in ['y', 'n']:
            print('请按指定格式输入！')
        else:
            break
    if f == 'n':
        return True
    else:
        return False


# In[7]:


def stop(ui, s):
    '''
    大循环终止条件
    '''
    if abs(max(ui) - min(ui)) < 0.1 * abs(max(s) - min(s)) or (
            (len(argrelmax(ui)[0]) == 0 and len(argrelmin(ui)[0]) == 0)):
        return '符合'
    else:
        return '不符合'


# In[8]:


# 与数据相关的函数
def getData_txt(filename):
    '''
    打开txt文件,文件结构必须是：两列,第一列是时间，第二列是幅值，两列之间至少空一格.
    '''
    t = []
    s = []
    with open(filename) as fp:
        for line in fp.readlines():
            linelst = line.split()
            t.append(float(linelst[0]))
            s.append(float(linelst[1]))
    return np.array(t), np.array(s)


def extension_data(t, s):
    '''
    扩展数据，将原始数据左边三分之一镜像放在数据左端点之前，将原始数据右边三分之一数据镜像放在数据右端点之后.
    该函数返回的是延拓好的数据.
    '''
    s = list(s)
    N = len(t)
    N1 = ceil(N / 3)
    N2 = floor(2 * N / 3)
    s_left = s[1:N1]
    s_left.reverse()
    s_right = s[N2:-1]
    s_right.reverse()
    t_left = list(-1 * t[1:N1])
    t_left.reverse()
    t_right = list(t[N2:-1] + (N - N2) * (t[1] - t[0]))
    t = list(t)
    S = s_left + s + s_right
    T = t_left + t + t_right
    return np.array(T), np.array(S)


# In[9]:


def plot_data_extensioned(t, s, PF, remnant):
    '''
    画图相关的函数,画经过extension_data函数延拓之后分解的结果的图像.
    '''
    fig = plt.figure(figsize=(16, 9))
    nfig = len(PF) + 2
    ax1 = fig.add_subplot(nfig, 1, 1)
    ax1.plot(t, s)
    N = len(t)
    N1 = ceil(N / 3)
    N11 = len(s[1:N1])
    N22 = len(PF[1]) - N11
    for i in range(len(PF)):
        axi = fig.add_subplot(nfig, 1, i + 2)
        axi.plot(t, PF[i][N11:N22])
    axn = fig.add_subplot(nfig, 1, nfig)
    axn.plot(t, remnant[N11:N22][-1])
    plt.tight_layout()


def plot_data_raw(t, s, PF, remnant, size, fontsize):
    '''
    画图相关的函数,画未经过extension_data函数延拓之后分解的结果.
    '''
    fig = plt.figure(figsize=size)
    nfig = len(PF) + 2
    ax1 = fig.add_subplot(nfig, 1, 1)
    ax1.plot(t, s)
    ax1.set_xlabel('t', fontsize=fontsize[0])
    ax1.set_ylabel('Data', fontsize=fontsize[0])
    ax1.tick_params(labelsize=fontsize[1])
    for i in range(len(PF)):
        axi = fig.add_subplot(nfig, 1, i + 2)
        axi.plot(t, PF[i])
        axi.set_xlabel('t', fontsize=fontsize[0])
        axi.set_ylabel('PF%d' % (i), fontsize=fontsize[0])
        axi.tick_params(labelsize=fontsize[1])
    axn = fig.add_subplot(nfig, 1, nfig)
    axn.plot(t, remnant[-1])
    axn.set_xlabel('t', fontsize=fontsize[0])
    axn.set_ylabel('remnant', fontsize=fontsize[0])
    axn.tick_params(labelsize=fontsize[1])
    plt.tight_layout()


# In[10]:


def write_data(t, A, S, PF, remnant, name):
    flag = 0
    dirname = name + '_lmdResult'
    if os.path.exists('./output_data/%s' % dirname):
        question = input('文件已经存在,你要覆盖吗?y/n ')
        while question not in ['y', 'n']:
            print('请按指定格式输入！')
            question = input('文件已经存在,你要覆盖吗?y/n ')
        if question == 'y':
            shutil.rmtree('./output_data/%s' % dirname)
        else:
            flag = 1

    if flag == 0:
        os.mkdir('./output_data/%s' % dirname)
        for i in range(len(PF)):
            with open('./output_data/{0}/ASPFR{1}.txt'.format(dirname, i), 'w') as file:
                for j in range(len(t)):
                    file.writelines(str('%-7.25f' % A[i][j]) + '\t' + str('%-7.17f' % S[i][j]) + '\t' + \
                                    str('%-7.17f' % PF[i][j]) + '\t' + str('%-7.17f' % remnant[i][j]) + '\n')

    print('文件已写入./output_data/%s' % dirname)


# In[11]:


def ma(t, s):
    '''
    局域均值、幅值函数.  
    '''
    xmax, ymax, xmin, ymin = extremum_coordinates_with_direct_extension(t, s)  # 极值坐标
    fup = Hermite(xmax, ymax).get_all_hermite_values(t)
    fdown = Hermite(xmin, ymin).get_all_hermite_values(t)
    print('mean(fup-fdown):', np.mean(fup - fdown) / 4)
    fup, fdown = disjoin(fup, fdown, t, e=np.mean(fup - fdown) / 4)
    m = (fup + fdown) / 2
    a = abs((fup - fdown) / 2)
    return m, a


# In[12]:


def lmd_main(t, s, simple='no', len_pf=0):
    '''
    lmd主函数.如果只做lmd,则默认simpl，len_pf即可，这两个参数是给elmd用的。
    '''
    # t,s=extension_data(t,s)#将数据左侧三分之一添加到左侧，右侧三分之一添加到右侧.
    ui = s
    if simple == 'no':
        N = len(t)
        A, S, PF, remnant, i = [], [], [], [], 0
        # while ask(t, ui):
        si = ui
        ai = np.ones(len(t))
        while True:
            m, a = ma(t, si)
            hi = si - m
            ai = ai * a
            si = hi / a
            stop_this = sum(abs(a - np.ones(N)) / N)
            print(stop_this)
            if stop_this < 30:
                break

        for j in range(len(si)):  # 将纯调频函数的上下界设为1
            if si[j] > 1:
                si[j] = 1
            elif si[j] < -1:
                si[j] = -1
        i += 1
        pfi = ai * si
        ui = ui - pfi
        A.append(ai)
        S.append(si)
        PF.append(pfi)
        remnant.append(ui)
        print('已产生{}个PF,残余项{}终止条件！'.format(i, stop(ui, s)))
        A = np.array(A)
        S = np.array(S)
        PF = np.array(PF)
        remnant = np.array(remnant)

    else:
        len_pf = len_pf
        N = len(t)
        A = np.zeros([len_pf, N])
        S = np.zeros([len_pf, N])
        PF = np.zeros([len_pf, N])
        remnant = np.zeros([len_pf, N])
        for i in range(len_pf):
            si = ui
            ai = np.ones(len(t))
            while True:
                m, a = ma(t, si)
                hi = si - m
                ai = ai * a
                si = hi / a
                print('h:', max(hi), min(hi), np.mean(hi))
                print('a:', max(a), min(a), np.mean(a))
                print('si:', max(si), min(si), np.mean(si))
                stop_this = sum(abs(a - np.ones(N)))
                print('stop_little:', stop_this)
                if stop_this < 30:
                    break

            plt.plot(si)
            for j in range(len(si)):  # 将纯调频函数的上下界设为1
                if si[j] > 1:
                    si[j] = 1
                elif si[j] < -1:
                    si[j] = -1
            pfi = ai * si
            ui = ui - pfi
            A[i] = ai
            S[i] = si
            PF[i] = pfi
            remnant[i] = ui
            print('已产生{}个PF.'.format(i + 1))
    print('\n')
    return A, S, PF, remnant


def elmd_main(t, s, sigma_gauss, n_gauss):
    '''
    elmd主函数
    '''
    N = len(t)
    sigma_source = np.sqrt(s.var())
    print('第1个lmd:')
    alpha = sigma_gauss / sigma_source
    np.random.seed(0)
    gaus_1 = alpha * np.random.normal(0, sigma_gauss, N)
    source_1 = s + gaus_1
    lmd_1 = lmd_main(t, source_1)
    len_pf = len(lmd_1[2])
    A = np.zeros([len_pf, N])
    S = np.zeros([len_pf, N])
    PF = np.zeros([len_pf, N])
    remnant = np.zeros([len_pf, N])
    A = A + lmd_1[0]
    S = S + lmd_1[1]
    PF = PF + lmd_1[2]
    remnant = remnant + lmd_1[3]
    n_gauss = n_gauss
    if n_gauss > 1:
        for i in range(1, n_gauss):
            print('第%d个lmd:' % (i + 1))
            np.random.seed(i)
            gaus_i = alpha * np.random.normal(0, sigma_gauss, N)
            source_i = s + gaus_i
            lmd_i = lmd_main(t, source_i, simple='yes', len_pf=len_pf)
            A = A + lmd_i[0]
            S = S + lmd_i[1]
            PF = PF + lmd_i[2]
            remnant = remnant + lmd_i[3]

        A = A / n_gauss
        S = S / n_gauss
        PF = PF / n_gauss
        remnant = remnant / n_gauss

    return A, S, PF, remnant
