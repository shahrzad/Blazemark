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


def create_dict(directories):
    thr=[]
    data_files=[]
    
    for directory in directories:
        [data_files.append(i) for i in glob.glob(directory+'/*.dat')]
    
    chunk_sizes=[]
    num_iterations=[]
    iter_lengths=[]
    nodes=[]
    
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
    
    d={}
    for node in nodes:
        d[node]={}
        for ni in num_iterations:              
            d[node][ni]={}           
            for th in thr:
                d[node][ni][th]={}
                for c in chunk_sizes:
                    d[node][ni][th][c]={}
                    for il in iter_lengths:
                        d[node][ni][th][c][il]={}
                                                           
    data_files.sort()   
    data=np.zeros((len(data_files),5))  
    i=0
    for filename in data_files:                
        f=open(filename, 'r')
                 
        result=f.readlines()
        avg=0
        if len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
            (node, _,_, th, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
            chunk_size=1
        else:
            (node, _,_, th, chunk_size, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
        chunk_size=int(chunk_size)        
        th=int(th)       
        iter_length=int(iter_length)
        num_iteration=int(num_iteration)    
        first=True
        for r in [r for r in result if r!='\n']:  
            if not first:
                avg+=float(r.split('in ')[1].split('microseconds')[0].strip())
            else:
                first=False
        d[node][num_iteration][th][chunk_size][iter_length]=avg/5
        data[i,:]=np.asarray([num_iteration, th, chunk_size,iter_length, avg])
        i=i+1
    return (data, d, thr, iter_lengths, num_iterations)  

marvin_dir='/home/shahrzad/repos/Blazemark/data/grain_size/marvin'
(data, d, thr, iter_lengths, num_iterations)=create_dict([marvin_dir])


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

                
                
grain_size=iter_lengths
i=1
for node in d.keys():
    for ni in d[node].keys():
        for th in d[node][ni].keys():        
            plt.figure(i)
            grain_sizes=[g for g in d[node][ni][th].keys()]
            results=[d[node][ni][th][g] for g in d[node][ni][th].keys()]
            plt.scatter(grain_sizes,results,marker='.',label='true')
            plt.xlabel('grain size')
            plt.ylabel('execution time')
            plt.xscale('log')
            plt.title('num iterations:'+str(ni)+'  '+str(th)+' threads')
            plt.axvline(ni/th,color='gray',linestyle='dotted')    
            i=i+1
            train_data=data[np.logical_and(data[:,0]==ni,data[:,1]==th)]
            param_bounds=([0,0,0,0,-np.inf,0],[np.inf,1,np.inf,np.inf,np.inf,np.inf])
    
            popt, pcov=curve_fit(my_func_g,train_data[:,:-1],train_data[:,-1],method='trf',bounds=param_bounds)
            z=my_func_g(train_data[:,:-1],*popt)
            plt.plot(grain_sizes,z,marker='.',label='pred')        
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
