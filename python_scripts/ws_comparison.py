#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:09:14 2020

@author: shahrzad
"""

import pandas
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import glob
import matplotlib.tri as mtri
from mpl_toolkits import mplot3d
import math
from scipy.optimize import curve_fit
from collections import Counter
from scipy.optimize import nnls
import csv
#from scipy.optimize import nnls
#from sklearn.metrics import r2_score

import grain_size_funcs as gf

filename_ws='/home/shahrzad/repos/Blazemark/data/normal_grain_data_perf_all.csv'
filename_nows='/home/shahrzad/repos/Blazemark/data/nows_grain_data_perf_all.csv'

titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'

for filename in [filename_nows, filename_ws]:
    dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
    for col in titles[1:]:
        dataframe[col] = dataframe[col].astype(float)
    
    node='marvin'
    nodes=dataframe['node'].drop_duplicates().values
    nodes.sort()
    node_selected=dataframe['node']==node
    iter_selected=dataframe['iter_length']==1
    th_selected=dataframe['num_threads']>=1
    
    df_n_selected=dataframe[node_selected & iter_selected & th_selected][titles[1:]]
    
    threads={}
    thr=df_n_selected['num_threads'].drop_duplicates().values
    thr.sort()
    threads[node]=thr
    
    problem_sizes=df_n_selected['problem_size'].drop_duplicates().values
    problem_sizes.sort()
    
    array=df_n_selected.values
    for ps in problem_sizes:
        if 'ws' in filename:
            label='work stealing off'
        else:
            label='work stealing on'
        array_ps=array[array[:,0]==ps]
        th=4
        array_t=array_ps[array_ps[:,2]==th]
        plt.axes([0, 0, 2, 2])

        plt.axvline(ps/th,color='lightgray',linestyle='dashed')
        plt.scatter(array_t[:,5], array_t[:,-1],label=label)   
        plt.xscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={"size":15})
        plt.xlabel('Grain size')
        plt.ylabel('Execution Time($\mu{sec}$)')
plt.savefig(perf_dir+'/thesis/'+str(int(ps))+'_'+str(int(th))+'_ws.png',bbox_inches='tight')

plt.savefig(perf_dir+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_ws_compared.png',bbox_inches='tight')

