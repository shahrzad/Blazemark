#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:19:59 2020

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


def create_dict(directories,to_csv=False):
    thr=[]
    data_files=[]
    
    for directory in directories:
        [data_files.append(i) for i in glob.glob(directory+'/*.dat')]
    
    chunk_sizes=[]
    num_iterations=[]
    iter_lengths=[]
    nodes=[]
    
    if to_csv:
        f_csv=open('/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv','w')
        f_writer=csv.writer(f_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time'])

    for filename in data_files:
        if len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
            (node, _,_, th, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
            chunk_size=1
        else:
            (node, _,_, th, chunk_size, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
        chunk_size=int(chunk_size)
        th=int(th)
        iter_length=int(iter_length)
        num_iteration=int(num_iteration)
        if node not in nodes:
            nodes.append(node)
        if iter_length not in iter_lengths:
            iter_lengths.append(iter_length)
        if num_iteration not in num_iterations:
            num_iterations.append(num_iteration)        
        if th not in thr:
            thr.append(th)
        if chunk_size not in chunk_sizes:
            chunk_sizes.append(chunk_size)

    nodes.sort()
    num_iterations.sort()
    iter_lengths.sort()
    thr.sort()              
    
#    d={}
#    for node in nodes:
#        d[node]={}
#        for ni in num_iterations:              
#            d[node][ni]={}           
#            for th in thr:
#                d[node][ni][th]={}
#                for c in chunk_sizes:
#                    d[node][ni][th][c]={}
#                    for il in iter_lengths:
#                        d[node][ni][th][c][il]={}
                                                           
    data_files.sort()   
    problem_sizes=[]
    
    for filename in data_files:                
        f=open(filename, 'r')
                 
        result=f.readlines()
        avg=0
        if len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
            (node, _,_, th, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
            chunk_size=1
        else:
            (node, _,_, th, chunk_size, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
        chunk_size=float(chunk_size)        
        th=float(th)       
        iter_length=float(iter_length)
        num_iteration=float(num_iteration)    
        first=True
        for r in [r for r in result if r!='\n']:  
            if not first:
                avg+=float(r.split('in ')[1].split('microseconds')[0].strip())
            else:
                first=False
#        d[node][num_iteration][th][chunk_size][iter_length]=avg/5
        problem_size=num_iteration*iter_length
        if problem_size not in problem_sizes:
            problem_sizes.append(problem_size)
        grain_size=chunk_size*iter_length
        num_tasks=np.ceil(num_iteration/chunk_size)
        L=np.ceil(num_tasks/th)
        w_c=L*grain_size
        if num_tasks%th==1 and num_iteration%chunk_size!=0:
            w_c=(L-1)*grain_size+(num_iteration%chunk_size)*grain_size
        f_writer.writerow([node,problem_size,num_iteration,th,chunk_size,iter_length,grain_size,w_c,num_tasks,avg/5])

    if to_csv:
        f_csv.close()
#    return (data, d, thr, iter_lengths, num_iterations)  

marvin_dir='/home/shahrzad/repos/Blazemark/data/grain_size/marvin'
create_dict([marvin_dir],1)


def my_func_g(ndata,alpha,gamma,d,h,q,ts): 
    kappa=0.
    N=ndata[:,1]
    n_t=ndata[:,0]
    cs=ndata[:,2]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    g=cs*ndata[:,3]
    w_c=L*g
    w_c[n_t%N==1]=((L-1)*g+(n_t%cs)*ndata[:,3])[n_t%N==1]
    return q*N+alpha*L+(ts+ts*(gamma)*(M-1)+ts*kappa*M*(M-1))*(w_c)+h*n_t*np.heaviside(n_t-N,1)+(d/N)*((n_t-1)%N)*np.heaviside(N-n_t,1)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)



titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'


dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)

nodes=dataframe['node'].drop_duplicates().values
nodes.sort()

problem_sizes=dataframe['problem_size'].drop_duplicates().values
problem_sizes.sort()


for node in nodes:
    node_selected=dataframe['node']==node
    df_n_selected=dataframe[node_selected][titles[1:]]
    
    thr=df_n_selected['num_threads'].drop_duplicates().values
    thr.sort()

    for ps in problem_sizes:
        ps_selected=dataframe['problem_size']==ps
        df_nps_selected=df_n_selected[ps_selected]
        
        array=df_nps_selected.values
        array=array.astype(float)
        
        data_size=int(np.shape(array)[0])
        print(int(ps),data_size)
        
        train_size=data_size#int(np.ceil(data_size*0.6))
        tes_size=data_size-train_size
        
        per = np.random.permutation(data_size)
        train_set=array[per[0:train_size],:-1] 
        train_labels=array[per[0:train_size],-1]  
        test_set=array[per[train_size:],:-1]  
        test_labels=array[per[train_size:],-1]  
        
        param_bounds=([0,0,0,0,-np.inf,0],[np.inf,1,np.inf,np.inf,np.inf,np.inf])
        popt, pcov=curve_fit(my_func_g,train_set,train_labels,method='trf',bounds=param_bounds)

        
        i=1
        for th in thr:
            new_array=train_set[train_set[:,2]==th]

            plt.figure(i)
#            z=my_func_g(train_data[:,:-1],*popt)

            plt.scatter(new_array[:,5],train_labels[train_set[:,2]==th],marker='.',label='true')
            plt.xlabel('grain size')
            plt.ylabel('execution time')
            plt.xscale('log')
            plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
            plt.axvline(ps/th,color='gray',linestyle='dotted')  
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            i=i+1
            
                


        i=1
        for th in thr:
            new_array=test_set[test_set[:,2]==th]

            plt.figure(i)
#            z=my_func_g(train_data[:,:-1],*popt)

            plt.scatter(new_array[:,5],test_labels[test_set[:,2]==th],marker='.',label='true')
            plt.xlabel('grain size')
            plt.ylabel('execution time')
            plt.xscale('log')
            plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
            plt.axvline(ps/th,color='gray',linestyle='dotted')  
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            i=i+1
            
                
