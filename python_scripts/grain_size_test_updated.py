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
from scipy.optimize import nnls


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
        if 'seq' in filename:
                (node, _,_, _, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
                chunk_size=0
                th=1      
        elif len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
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
    marvin_dir='/home/shahrzad/repos/Blazemark/data/grain_size/marvin'
    excludes=[marvin_dir+'/marvin_grain_size_1_1_2000_50000.dat',marvin_dir+'/marvin_grain_size_1_1_1000_100000.dat']
    
    for filename in data_files:   
        if filename not in excludes:             
            f=open(filename, 'r')
                     
            result=f.readlines()
            avg=0
            if 'seq' in filename:
                (node, _,_, _, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
                chunk_size=0
                th=1                
            elif len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
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
            avg=avg/(len([r for r in result if r!='\n'])-1)
    #        d[node][num_iteration][th][chunk_size][iter_length]=avg/5
            problem_size=num_iteration*(iter_length)
            if problem_size not in problem_sizes:
                problem_sizes.append(problem_size)
            grain_size=chunk_size*(iter_length)
            if chunk_size!=0:
                num_tasks=np.ceil(num_iteration/chunk_size)
                L=np.ceil(num_tasks/th)
                w_c=L*grain_size
                if th==1:
                    w_c=num_iteration*(iter_length)
                if num_tasks%th==1 and num_iteration%chunk_size!=0:
#                    w_c_1=problem_size+(1-th)*(L-1)*grain_size
                    w_c=(L-1)*grain_size+(num_iteration%chunk_size)*(iter_length)
            else:
                num_tasks=0
                L=0
                w_c=0
            
            
            f_writer.writerow([node,problem_size,num_iteration,th,chunk_size,iter_length,grain_size,w_c,num_tasks,avg])

    if to_csv:
        f_csv.close()
#    return (data, d, thr, iter_lengths, num_iterations)  

marvin_dir='/home/shahrzad/repos/Blazemark/data/grain_size/marvin'
medusa_dir='/home/shahrzad/repos/Blazemark/data/grain_size/medusa'
results_dir='/home/shahrzad/repos/Blazemark/results/grain_size'
create_dict([results_dir],1)

create_dict([marvin_dir,medusa_dir],1)


def my_func_g_1(ndata,alpha,gamma,d,h,q): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    return q*(N)+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d*(N-1)/N)*((n_t-1))*np.heaviside(N-n_t,1)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)


def my_func_g_2(ndata,alpha,gamma,d,h,q): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d*(1)/N)*((n_t-1))*np.heaviside(N-n_t,1)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)

def my_func_g_3(ndata,alpha,gamma,d,h,q): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d/(ps*N))*((n_t-1))*np.heaviside(N-n_t,1)
#
#def my_func_g_3(ndata,alpha,q):
#    N=ndata[:,2]
#    n_t=ndata[:,-1]
#    M=np.minimum(n_t,N) 
#    L=np.ceil(n_t/(N))
#    w_c=ndata[:,-2]
#    ps=ndata[:,0]
#    [gamma,d,h]=[1.08978362e-02, 5.40650289e+02, 5.42282886e-02]
#    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d/(ps*N))*((n_t-1))*np.heaviside(N-n_t,1)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)
#

def my_func_g_4(ndata,alpha,gamma,d,h,q): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t/(ps*M)+(d/ps)*(n_t-1)*np.heaviside(N-n_t,1)

#    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t/(ps*M)+(d*ps)*(n_t**2-1)*np.heaviside(N-n_t,1)


def my_func_g_4_part1(ndata,alpha,gamma,d,h,q): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=ndata[:,0]

    return d*((ps%w_c)*(n_t)+(w_c-ps%w_c)*(n_t-1))*np.heaviside(N-n_t,0)

def my_func_g_4_part2(ndata,alpha,gamma,d,h,q): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=ndata[:,0]

    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t/(ps*M)

def my_func_g_5(ndata,alpha,gamma,d,h,q): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t/(M)+d*((ps%w_c)*(n_t)+(w_c-ps%w_c)*(n_t-1))*np.heaviside(N-n_t,0)

def my_func_g_6(ndata,alpha,gamma,h,q): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
#    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d/n_t)*(N-n_t)*((N-n_t-1))*np.heaviside(N-n_t,1)
    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t/(ps*M)
def my_func_g_7(ndata,alpha,gamma,h,q): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
#    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d/n_t)*(N-n_t)*((N-n_t-1))*np.heaviside(N-n_t,1)
    return q*(N-1)*(N-2)/ps+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*np.heaviside(n_t-N,1)+h*(w_c-ps/M)/(ps/M)#(h)*(N-n_t)*(n_t)*np.heaviside(N-n_t,1)


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

i=1   
popt={}                
for node in nodes:
    np.random.seed(0)                

    node_selected=dataframe['node']==node
    nt_selected=dataframe['num_tasks']!=0
    iter_selected=dataframe['iter_length']==1

    df_n_selected=dataframe[node_selected & nt_selected & iter_selected][titles[1:]]
    
    thr=df_n_selected['num_threads'].drop_duplicates().values
    thr.sort()

    problem_sizes=df_n_selected['problem_size'].drop_duplicates().values
    problem_sizes.sort()



    array=df_n_selected.values
    array=array.astype(float)
    
    per=np.array([])
    for th in thr:
        ind_th=np.where(array[:,2]==th)[0]
        data_size=int(1*np.shape(ind_th)[0])
        per = np.concatenate((ind_th[np.random.permutation(data_size)],per),axis=0)
    train_indices=per.astype(int)
                     
    train_size=np.shape(train_indices)[0]
    print(train_size)
    
    train_set=array[train_indices,:-1] 
    train_labels=array[train_indices,-1]  
    
    all_indices=np.arange(np.shape(array)[0])
    
    test_indices=[ind for ind in all_indices if ind not in train_indices]
    test_set=array[test_indices,:-1]  
    test_labels=array[test_indices,-1]  
    print(np.shape(test_set)[0])


    param_bounds=([0,0,0,0,0],[np.inf,1,np.inf,np.inf,np.inf])
#    param_bounds=([0,-np.inf],[np.inf,np.inf])

#        param_bounds=([0,-np.inf],[np.inf,np.inf])

#    popt_1, pcov=curve_fit(my_func_g_1,train_set,train_labels,method='trf',bounds=param_bounds)
#    popt_2, pcov=curve_fit(my_func_g_2,train_set,train_labels,method='trf',bounds=param_bounds)
    popt_3, pcov=curve_fit(my_func_g_3,train_set,train_labels,method='trf',bounds=param_bounds)
#    param_bounds=([0,0,0,0,-np.inf,-np.inf,-np.inf],[np.inf,1,np.inf,np.inf,np.inf,np.inf,np.inf])
    popt_5, pcov=curve_fit(my_func_g_5,train_set,train_labels,method='trf',bounds=param_bounds)
    popt_4, pcov=curve_fit(my_func_g_4,train_set,train_labels,method='trf',bounds=param_bounds)

    param_bounds=([0,0,0,-np.inf],[np.inf,1,np.inf,np.inf])

    popt_6, pcov=curve_fit(my_func_g_6,train_set,train_labels,method='trf',bounds=param_bounds)
    popt_7, pcov=curve_fit(my_func_g_7,train_set,train_labels,method='trf',bounds=param_bounds)

    popt[node]=popt_3
    
        
    

    for ps in [1e4,2e5,5e6,1e7]:#problem_sizes[-200:]:
        
        array_ps=train_set[train_set[:,0]==ps]
        labels_ps=train_labels[train_set[:,0]==ps]
        
        a_s=np.argsort(array_ps[:,5])
        for ir in range(np.shape(array_ps)[1]):
            array_ps[:,ir]=array_ps[a_s,ir]
        labels_ps=labels_ps[a_s] 

#        param_bounds=([0,0,0,0,-np.inf],[np.inf,1,np.inf,np.inf,np.inf])
#        popt_4, pcov=curve_fit(my_func_g_4,train_set,train_labels,method='trf',bounds=param_bounds)
#        param_bounds=([0,0,0,-np.inf],[np.inf,1,np.inf,np.inf])
#
#        popt_6, pcov=curve_fit(my_func_g_6,array_ps,labels_ps,method='trf',bounds=param_bounds)
#        popt_7, pcov=curve_fit(my_func_g_7,array_ps,labels_ps,method='trf',bounds=param_bounds)

        for th in thr:
            new_array=array_ps[array_ps[:,2]==th]
            new_labels=labels_ps[array_ps[:,2]==th]
            if np.shape(new_array[new_array[:,3]>0])[0]>10:
                plt.figure(i)
#                z_1=my_func_g_1(new_array,*popt_1)
#                z_2=my_func_g_2(new_array,*popt_2)
                z_3=my_func_g_3(new_array,*popt_3)
                z_5=my_func_g_5(new_array,*popt_5)
                z_6=my_func_g_6(new_array,*popt_6)
                z_7=my_func_g_7(new_array,*popt_7)

                z_4=my_func_g_4(new_array,*popt_4)
                z_4_1=my_func_g_4_part1(new_array,*popt_5)
                z_4_2=my_func_g_4_part2(new_array,*popt_4)
#                plt.scatter(new_array[:,5][new_array[:,5]>=ps/th],(new_labels/1000-z_4_2/1000)[new_array[:,5]>=ps/th],marker='.',label='true')

                plt.scatter(new_array[:,5],new_labels,marker='.',label='true')
##                plt.scatter(new_array[:,5],z_1,marker='.',label='pred1')
##                plt.scatter(new_array[:,5],z_2,marker='.',label='pred2')
                plt.scatter(new_array[:,5],z_3,marker='.',label='pred3')
#                plt.scatter(new_array[:,5][new_array[:,5]>=ps/th],new_array[:,-1][new_array[:,5]>=ps/th],marker='.',label='pred4')

                plt.scatter(new_array[:,5],z_4_1,marker='.',label='part1')
#                plt.scatter(new_array[:,5],z_4_2,marker='.',label='part2')

#                plt.scatter(new_array[:,5],z_5,marker='.',label='pred5')
#                plt.scatter(new_array[:,5],z_7,marker='.',label='pred7')

#                plt.scatter(new_array[:,5],z_4,marker='.',label='pred4')
        
                plt.xlabel('grain size')
                plt.ylabel('execution time')
                plt.xscale('log')
                plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
                plt.axvline(ps/th,color='gray',linestyle='dotted')  
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#               plt.save fig(perf_dir+node+'/'+str(int(ps))+'_'+str(int(th))+'.png',bbox_inches='tight')

                i=i+1        
                
                
#################################
#all the data together
#################################

i=1   
            
np.random.seed(0)                

dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)

nodes=dataframe['node'].drop_duplicates().values
nodes.sort()

problem_sizes=dataframe['problem_size'].drop_duplicates().values
problem_sizes.sort()

i=1   
popt={}       


train_set_all=np.empty(shape=(0,8))
train_labels_all=np.array([])

test_set_all={}
test_labels_all={}
 
train_lengths={}
test_lengths={}

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
    
    per=np.array([])

    for th in thr:
        ind_th=np.where(array[:,2]==th)[0]
        data_size=int(0.6*np.shape(ind_th)[0])
        per = np.concatenate((ind_th[np.random.permutation(data_size)],per),axis=0)
    train_indices=per.astype(int)
                     
    train_size=np.shape(train_indices)[0]

    train_set=array[train_indices,:-1] 
    train_label=array[train_indices,-1]  
    
    all_indices=np.arange(np.shape(array)[0])
    
    test_indices=[ind for ind in all_indices if ind not in train_indices]
    test_set=array[test_indices,:-1]  
    test_label=array[test_indices,-1]  
    test_size=np.shape(test_set)[0]

    train_set_all=np.concatenate((train_set_all,train_set),0)
    train_labels_all=np.concatenate((train_labels_all,train_label))

    test_set_all[node]=test_set
    test_labels_all[node]=test_label
    
    train_lengths[node]=train_size
    test_lengths[node]=test_size
    


param_bounds=([0,0,0,0,-np.inf],[np.inf,1,np.inf,np.inf,np.inf])
#    param_bounds=([0,-np.inf],[np.inf,np.inf])

#        param_bounds=([0,-np.inf],[np.inf,np.inf])

#    popt_1, pcov=curve_fit(my_func_g_1,train_set,train_labels,method='trf',bounds=param_bounds)
#    popt_2, pcov=curve_fit(my_func_g_2,train_set,train_labels,method='trf',bounds=param_bounds)
popt_3, pcov=curve_fit(my_func_g_3,train_set_all,train_labels_all,method='trf',bounds=param_bounds)
#    param_bounds=([0,0,0,0,-np.inf,-np.inf,-np.inf],[np.inf,1,np.inf,np.inf,np.inf,np.inf,np.inf])

#    popt_4, pcov=curve_fit(my_func_g_4,train_set,train_labels,method='trf',bounds=param_bounds)
popt[node]=popt_3

    

for node in nodes:    
    for ps in problem_sizes:        
        array_ps=test_set_all[node][test_set_all[node][:,0]==ps]
        labels_ps=test_labels_all[node][test_set_all[node][:,0]==ps]
        if np.shape(array_ps)[0]>0:
            for th in thr:
                new_array=array_ps[array_ps[:,2]==th]
                new_labels=labels_ps[array_ps[:,2]==th]
                if np.shape(new_array[new_array[:,3]>0])[0]>10:
                    plt.figure(i)
    #                z_1=my_func_g_1(new_array,*popt_1)
    #                z_2=my_func_g_2(new_array,*popt_2)
                    z_3=my_func_g_3(new_array,*popt_3)
    #                z_4=my_func_g_4(new_array,*popt_4)
            
                    plt.scatter(new_array[:,5],new_labels,marker='.',label='true')
    #                plt.scatter(new_array[:,5],z_1,marker='.',label='pred1')
    #                plt.scatter(new_array[:,5],z_2,marker='.',label='pred2')
                    plt.scatter(new_array[:,5],z_3,marker='.',label='pred')
    #                plt.scatter(new_array[:,5],z_4,marker='.',label='pred4')
            
                    plt.xlabel('grain size')
                    plt.ylabel('execution time')
                    plt.xscale('log')
                    plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
                    plt.axvline(ps/th,color='gray',linestyle='dotted')  
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    plt.savefig(perf_dir+node+'/'+str(int(ps))+'_'+str(int(th))+'.png',bbox_inches='tight')
    
                    i=i+1                    