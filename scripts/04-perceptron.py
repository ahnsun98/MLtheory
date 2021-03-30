# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

# multivariate_normal(평균,표준편차,사이즈): 정규분포를 다차원 공간에 대해 확장한 분포인 다변량 정규 분포에서 무작위 표본을 추출
from numpy.random import multivariate_normal


N1 = 20
Mu1 = [15,10]

N2 = 30
Mu2 = [0,0]

Variances = [15,30]

# 트레이닝 셋 데이터 무작위 생성
def prepare_dataset(variance):
    cov1 = np.array([[variance,0],[0,variance]])
    cov2 = np.array([[variance,0],[0,variance]])
    print('cov1\n',cov1,'\ncov2\n',cov2)
    
    df1 = DataFrame(multivariate_normal(Mu1,cov1,N1),columns=['x','y'])
    df1['type'] = 1
    #print('t=1\n',df1)
    df2 = DataFrame(multivariate_normal(Mu2,cov2,N2),columns=['x','y'])
    df2['type'] = -1
    #print('t=-1\n',df2)
    df = pd.concat([df1,df2],ignore_index=True)
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
    print('df\n',df)
    return df

def run_simulation(variance, data_graph, param_graph):
    train_set = prepare_dataset(variance)
    train_set1 = train_set[train_set['type']==1]
    train_set2 = train_set[train_set['type']==-1]
    ymin, ymax = train_set.y.min()-5, train_set.y.max()+10
    xmin, xmax = train_set.x.min()-5, train_set.x.max()+10
    data_graph.set_ylim([ymin-1, ymax+1])
    data_graph.set_xlim([xmin-1, xmax+1])
    data_graph.scatter(train_set1.x, train_set1.y, marker='o')
    data_graph.scatter(train_set2.x, train_set2.y, marker='x')

    # 파라미터 값 초기 설정 W=0
    w0 = w1 = w2 = 0.0
    # 바이어스 값 = 트레이닝 셋에 포함된 모든 x와 y의 평균값 <- 적절하게 바꾸면 알고리즘 수렴속도 빠르게 변화 가능
    #               분류 직선이 원점을 지나는 경우 바이어스 값이 1이어도 빠르게 수렴 
    bias = 0.5 * (train_set.x.mean() + train_set.y.mean())

    paramhist = DataFrame([[w0,w1,w2]], columns=['w0','w1','w2'])
    # 30번 반복으로 임의설정
    for i in range(30):
        # DataFrame 행을 (인덱스, 시리즈) 쌍으로 반복 ; 행으로 묶어 표현
        for index, point in train_set.iterrows():
            #print('index\n',index,'\npoint\n',point)
            x, y, type = point.x, point.y, point.type
            # 파라미터 수정
            if type * (w0*bias + w1*x + w2*y) <= 0:
                w0 += type * 1 
                w1 += type * x
                w2 += type * y
        paramhist = paramhist.append(
                        Series([w0,w1,w2], ['w0','w1','w2']),
                        ignore_index=True)
    err = 0
    for index, point in train_set.iterrows():
        x, y, type = point.x, point.y, point.type
        if type * (w0*bias + w1*x + w2*y) <= 0:
            err += 1
    # 제대로 분류되지 않은 데이터의 비율
    err_rate = err * 100 / len(train_set)

    linex = np.arange(xmin-5, xmax+5)
    liney = - linex * w1 / w2 - bias * w0 / w2
    label = "ERR %.2f%%" % err_rate
    data_graph.plot(linex,liney,label=label,color='red')
    data_graph.legend(loc=1)
    paramhist.plot(ax=param_graph)
    param_graph.legend(loc=1)

    
if __name__ == '__main__':
    fig = plt.figure()
    
    for c, variance in enumerate(Variances):
        print('c=',c,'variance=',variance)
        subplots1 = fig.add_subplot(2,2,c+1) # data_graph
        subplots2 = fig.add_subplot(2,2,c+2+1) # param_graph
        run_simulation(variance, subplots1, subplots2)
    fig.show()
    # 완전히 분류된 경우 파라미터의 변화가 도중에 멈추지만 완전히 분류되지 못한 경우 한없이 변화한다
