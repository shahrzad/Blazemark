#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:35:23 2020

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
from scipy.optimize import nnls

titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'
perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hxp_for_loop/1/all/'


dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)

nodes=dataframe['node'].drop_duplicates().values
nodes.sort()

problem_sizes=dataframe['problem_size'].drop_duplicates().values
problem_sizes.sort()


def find_flat(x,y):
    y_prev=y[0]
    prev_i=0
    regions=[]
    for i in range(1,np.shape(x)[0]):
        if abs(y[i]-y_prev)>0.04*y_prev:
            if i-prev_i>5:
                regions.append([x[prev_i],x[i]])
            prev_i=i

        y_prev=y[i]
    return regions
        

i=1
def my_func_g_3(ndata,alpha,gamma,d,h,q): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d/(ps*N))*((n_t-1))*np.heaviside(N-n_t,1)

for node in nodes:
    np.random.seed(0)                

    node_selected=dataframe['node']==node
    df_n_selected=dataframe[node_selected][titles[1:]]
    
    thr=df_n_selected['num_threads'].drop_duplicates().values
    thr.sort()

    problem_sizes=df_n_selected['problem_size'].drop_duplicates().values
    problem_sizes.sort()

    array=df_n_selected.values
    array=array.astype(float)
    for th in thr:
        all_regions={}

        new_array=array[array[:,2]==th]
        new_labels=array[array[:,2]==th]
            
        for ps in problem_sizes[-30:]:    
            
            array_ps=new_array[new_array[:,0]==ps][:,:-1]
            labels_ps=new_labels[new_labels[:,0]==ps][:,-1]
            
            a_s=np.argsort(array_ps[:,5])
            for ir in range(np.shape(array_ps)[1]):
                array_ps[:,ir]=array_ps[a_s,ir]
            labels_ps=labels_ps[a_s]    
            
            n_t=array_ps[:,-1]
            M=np.minimum(n_t,th) 
            L=np.ceil(n_t/th)
            w_c=array_ps[:,-2]
            prs=array_ps[:,0]
            
            param_bounds=([0,0,0,0,-np.inf],[np.inf,1,np.inf,np.inf,np.inf])
            popt_3, pcov=curve_fit(my_func_g_3,array_ps,labels_ps,method='trf',bounds=param_bounds)
            [alpha,gamma,d,h,q]=popt_3
#
#            plt.plot(array_ps[:,5],(th-1)*(th-2)*q/prs,label='q*(n-1)*(n-2)/ps')
#            plt.plot(array_ps[:,5],alpha*L,label='alpha*L')
#            plt.plot(array_ps[:,5],(1+(gamma)*(M-1))*(w_c),label='(1+(gamma)*(M-1))*(w_c)')
#            plt.plot(array_ps[:,5],h*n_t*(th-1)*np.heaviside(n_t-th,1),label='h*n_t*(N-1)*np.heaviside(n_t-N,1)')
#            plt.plot(array_ps[:,5],(d/(ps*th))*((n_t-1))*np.heaviside(th-n_t,1),label='q*(n-1)*(n-2)/ps')
#            plt.xscale('log')
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
#            plt.plot(array_ps[:,5],alpha*L+(1+(gamma)*(M-1))*(w_c),label='alpha*L')
#            plt.xscale('log')
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
#            plt.scatter(array_ps[:,5],w_c/ps,label='w_c/ps',marker='.')
#            plt.xscale('log')
#            plt.grid(True,'both')
#
#            
#            plt.scatter(array_ps[:,5],w_c,label='w_c',marker='.')
#            plt.plot(array_ps[:,5],L,label='L')
##            plt.plot(array_ps[:,5],(n_t),label='(n_t)')
#            plt.axvline(np.sqrt(alpha*ps/(0.1*th)),color='purple')
#            plt.axvline((ps/(th*(th+1)))+(0.01/(th+1)),color='green')
#
#            plt.plot(array_ps[:,5],alpha*L,label='alpha*L')
#           
##            plt.plot(array_ps[:,5],alpha*ps/(th*(array_ps[:,5]**2)),label='alpha*ps/(g*th)')
#
#            plt.xscale('log')
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            
            if np.shape(np.unique(array_ps[:,5]))[0]>20 and np.shape(array_ps)[0]>10:
#                print(ps)
                plt.figure(i)
                plt.scatter(array_ps[:,5], labels_ps,marker='.',label='ps:'+str(int(ps)))   
                plt.scatter(array_ps[:,5], my_func_g_3(array_ps,*popt_3),marker='.',label='fit')   
                plt.axvline((ps/(th*(th+1)))+(0.01/(th+1)),color='green')
                plt.axvline(np.sqrt(alpha*ps/(0.1*th)),color='purple')
                plt.xlabel('grain size')
                plt.ylabel('execution time')
                plt.xscale('log')
                plt.title(str(th)+' threads')
                plt.grid(True,'both')
#                plt.axvline(ps/th,color='gray',linestyle='dotted')  
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#        
                all_regions[ps]=find_flat(array_ps[:,-1], labels_ps)
                plt.axvline(all_regions[ps][0][0],color='green')
                plt.axvline(all_regions[ps][-1][1],color='green')
#                if len(all_regions[ps])>0:
#                    for j in range(len(all_regions[ps])):
#                        plt.axvline(all_regions[ps][j][0],color='green')
#                        plt.axvline(all_regions[ps][j][1],color='green')
                i=i+1
                
                
ps1=1e7
ps2=4e6

array_ps1=new_array[new_array[:,0]==ps1][:,:-1]
labels_ps1=new_labels[new_labels[:,0]==ps1][:,-1]

a_s=np.argsort(array_ps1[:,5])
for ir in range(np.shape(array_ps1)[1]):
    array_ps1[:,ir]=array_ps1[a_s,ir]
labels_ps1=labels_ps1[a_s] 


array_ps2=new_array[new_array[:,0]==ps2][:,:-1]
labels_ps2=new_labels[new_labels[:,0]==ps2][:,-1]

a_s=np.argsort(array_ps2[:,5])
for ir in range(np.shape(array_ps2)[1]):
    array_ps2[:,ir]=array_ps2[a_s,ir]
labels_ps2=labels_ps2[a_s]     

set1=set(array_ps1[:,5].tolist())
set2=set(array_ps2[:,5].tolist())
all_gs=[s for s in set1.intersection(set2)]
all_gs.sort()
all_gs=np.asarray(all_gs)

plt.figure(i)
for j in np.arange(np.shape(all_gs)[0]):
    plt.scatter(all_gs[j],labels_ps1[np.where(array_ps1[:,5]==all_gs[j])[0][0]]-labels_ps2[np.where(array_ps2[:,5]==all_gs[j])[0][0]])
#plt.scatter(all_gs, labels_ps1-labels_ps2,marker='.',label='ps:'+str(int(ps)))   
plt.xlabel('grain size')
plt.ylabel('execution time')
plt.xscale('log')
plt.title(str(th)+' threads')
plt.grid(True,'both')
plt.axvline(ps/th,color='gray',linestyle='dotted')  
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            