#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:59:06 2021

@author: shahrzad
"""


import grain_size_funcs as gf
import numpy as np
from matplotlib import pyplot as plt
import pandas
from scipy.optimize import curve_fit
import math
from sklearn.metrics import r2_score
import blaze_funcs as bf

import matplotlib
matplotlib.rcParams.update({'font.size': 20})

medusa_gdir='/home/shahrzad/repos/Blazemark/data/final/grain_size/medusa/general/'
popt=gf.find_model_parameters([medusa_gdir])
node='medusa'

medusa_benchmark_dir='/home/shahrzad/repos/Blazemark/data/grain_size_benchmarks/blaze_runs/'
filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_bench_old.csv'

medusa_benchmark_dir='/home/shahrzad/repos/Blazemark/data/grain_size_benchmarks/grain_size/'
filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_bench_grain.csv'

medusa_benchmark_dir='/home/shahrzad/repos/Blazemark/data/grain_size_benchmarks/2/'
filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_bench.csv'

gf.create_dict_benchmark([medusa_benchmark_dir],filename)


titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']

dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)
    
node_selected=dataframe['node']=='medusa'
df_n_selected=dataframe[titles[1:]]
thr=df_n_selected['num_threads'].drop_duplicates().values
problem_sizes=df_n_selected['problem_size'].drop_duplicates().values

thr.sort()
problem_sizes.sort()

array=df_n_selected.values
i=1

        
def my_func(ndata,ts,alpha,gamma): 
    N=ndata[:,2]
    n_t=ndata[:,-2]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-3]
    ps=ndata[:,0]
    return alpha*L+ts*(1+(gamma)*(M-1))*(w_c)/ps


for ps in problem_sizes:
    array_ps=array[array[:,0]==ps]
    array_1=array_ps[array_ps[:,2]==1]
    ts=array_1[array_1[:,3]==1][0,-1]
    for th in thr:
        plt.figure(i)
        # plt.title("array size: "+str(int(ps/2))+", "+str(int(th))+" threads\n")
        plt.title("array size: "+str(int(ps/2))+"\n")
        array_t=array_ps[array_ps[:,2]==th]
        # z_5=my_func(array_t,ts,*popt[node])
        plt.scatter(array_t[:,5],array_t[:,-1], label=str(int(th))+" threads")   
        # plt.scatter(array_t[:,5],z_5,label='pred')   
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xscale('log')
        plt.xlabel('grain size')
        plt.ylabel('execution time($\mu{sec}$)') 
        i=i+1


for ps in problem_sizes:
    array_ps=array[array[:,0]==ps]
    array_1=array_ps[array_ps[:,2]==1]
    ts=array_1[array_1[:,3]==1][0,-1]
    for g in [2, 200, 2000, 20000, 200000]:
        plt.figure(i)
        plt.title("array size: "+str(int(ps/2))+", chunk size:"+str(int(g/2))+'\n')
        array_g=array_ps[array_ps[:,5]==g]
        plt.scatter(array_g[:,2],array_g[:,-1])   
        # plt.scatter(array_t[:,5],z_5,label='pred')   
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel('threads')
        plt.ylabel('execution time($\mu{sec}$)')    
        plt.xticks(thr)
        i=i+1