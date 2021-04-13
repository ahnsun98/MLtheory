# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import normal
from scipy.stats import norm


if __name__ == '__main__':

    # 실제 분포 -> training set 얻어냄
    mu_true = 2.0
    beta_true = 1.0

    # 사전 분포
    mu_0 = -1.0
    beta_0 = 1.0

    fig1 = plt.figure()
    fig2 = plt.figure()
    ds = normal(loc=mu_true, scale=1.0/beta_true, size=100)

    for c, n in enumerate([2,4,10,100]): # training set data 수 (N)
        trainset = ds[0:n]
        mu_ML = np.mean(trainset)
        mu_N = (beta_true*mu_ML + beta_0*mu_0/n)/(beta_true+beta_0/n)
        beta_N = beta_0 + n*beta_true

    # 평균μ에 대한 베이즈 추정 결과 표시
        subplot = fig1.add_subplot(2,2,c+1)
        subplot.set_title("N=%d" % n)
        linex = np.arange(-10,10.1,0.01)

        # 평균μ 확률 분포
        sigma = 1.0/beta_N
        mu_est = norm(loc=mu_N, scale=np.sqrt(sigma))
        label = "mu_N=%.2f\nvar=%.2f" % (mu_N, sigma)
        subplot.plot(linex, mu_est.pdf(linex), color='red', label=label)
        subplot.legend(loc=2)

        # training set 표시
        subplot.scatter(trainset, [0.2]*n, marker='o', color='blue')
        subplot.set_xlim(-5,5)
        subplot.set_ylim(0)


    # 다음에 얻을 수 있는 data의 추정 분포를 표시
        subplot = fig2.add_subplot(2,2,c+1)
        subplot.set_title("N=%d" % n)
        linex = np.arange(-10,10.1,0.01)

        # 실제 분포 표시
        orig = norm(loc=mu_true, scale=np.sqrt(1.0/beta_true))
        subplot.plot(linex, orig.pdf(linex), color='green', linestyle='--')

        # 추정 분포 표시
        sigma = 1.0/beta_true+1.0/beta_N
        mu_est = norm(loc=mu_N, scale=np.sqrt(sigma))
        label = "mu_N=%.2f\nvar=%.2f" % (mu_N, sigma)
        subplot.plot(linex, mu_est.pdf(linex), color='red', label=label)
        subplot.legend(loc=2)

        # training set 표시
        subplot.scatter(trainset, orig.pdf(trainset), marker='o', color='blue')
        subplot.set_xlim(-5,5)
        subplot.set_ylim(0)

    fig1.show()
    fig2.show()
