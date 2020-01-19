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
    marvin_dir='/home/shahrzad/repos/Blazemark/data/grain_size/marvin'
    excludes=[marvin_dir+'/marvin_grain_size_1_1_2000_50000.dat',marvin_dir+'/marvin_grain_size_1_1_1000_100000.dat']
    
    for filename in data_files:   
        if filename not in excludes:             
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
            avg=avg/(len([r for r in result if r!='\n'])-1)
    #        d[node][num_iteration][th][chunk_size][iter_length]=avg/5
            problem_size=num_iteration*iter_length
            if problem_size not in problem_sizes:
                problem_sizes.append(problem_size)
            grain_size=chunk_size*iter_length
            num_tasks=np.ceil(num_iteration/chunk_size)
            L=np.ceil(num_tasks/th)
            w_c=L*grain_size
            if th==1:
                w_c=ps
            if num_tasks%th==1 and num_iteration%chunk_size!=0:
                w_c=(L-1)*grain_size+(num_iteration%chunk_size)*iter_length
            f_writer.writerow([node,problem_size,num_iteration,th,chunk_size,iter_length,grain_size,w_c,num_tasks,avg])

    if to_csv:
        f_csv.close()
#    return (data, d, thr, iter_lengths, num_iterations)  

marvin_dir='/home/shahrzad/repos/Blazemark/data/grain_size/marvin'
create_dict([marvin_dir],1)


def my_func_g(ndata,alpha,gamma,d,h,q): 
    kappa=0.
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    ts=ndata[:,0]
    return q*N+alpha*L+(1+(gamma)*(M-1))*(w_c)+h*n_t*(N-1)*np.heaviside(n_t-N,1)+(d*(N-1)/N)*((n_t-1))*np.heaviside(N-n_t,1)#+q*(n_t-1)*(g-mflop%g)/g*np.heaviside(N-n_t,1)#+d1*(ts)*(g-(mflop%g))*(1/mflop)

def my_func_g(ndata,alpha,q): 
    kappa=0.
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    ts=ndata[:,0]
    return q+alpha*L+(w_c)



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
        
        train_size=int(np.ceil(data_size*0.6))
        tes_size=data_size-train_size
        
        per = np.random.permutation(data_size)
        train_set=array[per[0:train_size],:-1] 
        train_labels=array[per[0:train_size],-1]  
        test_set=array[per[train_size:],:-1]  
        test_labels=array[per[train_size:],-1]  
        
        param_bounds=([0,0,0,0,-np.inf],[np.inf,1,np.inf,np.inf,np.inf])
#        param_bounds=([0,-np.inf],[np.inf,np.inf])

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
            z=my_func_g(new_array,*popt)

            plt.scatter(new_array[:,5],test_labels[test_set[:,2]==th]/1e6,marker='.',label='true')
            plt.scatter(new_array[:,5],z/1e6,marker='.',label='pred')

            plt.xlabel('grain size')
            plt.ylabel('execution time')
            plt.xscale('log')
            plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
            plt.axvline(ps/th,color='gray',linestyle='dotted')  
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            i=i+1
            
                

        
        
def find_fit(ndata, labels, ncols=5):
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    Q[:,0]=L
    Q[:,1]=(M-1)*w_c
    Q[:,2]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
    Q[:,3]=n_t*np.heaviside(n_t-N,1)
    Q[:,4]=N

    m,_ = nnls(Q, labels-w_c)
    return m

def find_val(ndata, model):
    ncols=model.size
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    Q[:,0]=L
    Q[:,1]=(M-1)*w_c
    Q[:,2]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
    Q[:,3]=n_t*np.heaviside(n_t-N,1)
    Q[:,4]=N
    return np.dot(Q,model)+w_c



model=find_fit(train_set,train_labels)        
i=1
for th in thr:
    new_array=test_set[test_set[:,2]==th]

    plt.figure(i)
    z1=find_val(new_array,model)
    z2=my_func_g(new_array,*popt)
    plt.scatter(new_array[:,5],ps/test_labels[test_set[:,2]==th],marker='.',label='true')
    plt.scatter(new_array[:,5],ps/z1,marker='.',label='pred')
    plt.scatter(new_array[:,5],ps/z2,marker='.',label='curve_fit')

    plt.xlabel('grain size')
    plt.ylabel('mflops')
    plt.xscale('log')
    plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
    plt.axvline(ps/th,color='gray',linestyle='dotted')  
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    i=i+1
    
i=1
for th in thr:
    new_array=test_set[test_set[:,2]==th]

    plt.figure(i)
    z1=find_val(new_array,model)
    z2=my_func_g(new_array,*popt)
    plt.scatter(new_array[:,5],test_labels[test_set[:,2]==th]/ps,marker='.',label='true')
    plt.scatter(new_array[:,5],z1/ps,marker='.',label='pred')
    plt.scatter(new_array[:,5],z2/ps,marker='.',label='curve_fit')

    plt.xlabel('grain size')
    plt.ylabel('execution time')
    plt.xscale('log')
    plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
    plt.axvline(ps/th,color='gray',linestyle='dotted')  
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    i=i+1
    
    
###################################################################################################
# taking number of threads out
def find_fit(ndata, labels, params=None,ncols=5):
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    

    if params is None:
        Q=np.zeros((np.shape(ndata)[0],ncols))
        Q[:,0]=L
        Q[:,1]=(M-1)*w_c
        Q[:,2]=(1)*((n_t-1))*np.heaviside(N-n_t,1)
        Q[:,3]=n_t*np.heaviside(n_t-N,1)
        Q[:,4]=N
        print('None')
        m,_ = nnls(Q, labels-w_c)
        return m
    else:
        Q=np.zeros((np.shape(ndata)[0],ncols-2))
        Q[:,0]=(M-1)*w_c
        Q[:,1]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
        Q[:,2]=n_t*np.heaviside(n_t-N,1)
        print('not None')
        m,_ = nnls(Q, labels-w_c-N*params[-1]-L*params[0])
        print(m)
        print([params[0],m[0],m[1],m[2],params[-1]])
        return np.asarray([params[0],m[0],m[1],m[2],params[-1]])
    
def find_val(ndata, model):
    ncols=model.size
    Q=np.zeros((np.shape(ndata)[0],ncols))
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(N))
    w_c=ndata[:,-2]
    Q[:,0]=L
    Q[:,1]=(M-1)*w_c
    Q[:,2]=(1-1/N)*((n_t-1))*np.heaviside(N-n_t,1)
    Q[:,3]=n_t*np.heaviside(n_t-N,1)
    Q[:,4]=N
    return np.dot(Q,model)+w_c


i=1
th=1
array_th=array[array[:,2]==th]
a_s=np.argsort(array_th[:,5])
array_th=array_th[a_s]
  
data_size=int(np.shape(array_th)[0])
print(int(ps),data_size)

train_size=data_size#int(np.ceil(data_size*0.6))
tes_size=data_size-train_size

per = np.random.permutation(data_size)
train_set=array_th[per[0:train_size],:-1] 
train_labels=array_th[per[0:train_size],-1]  
test_set=array_th[per[train_size:],:-1]  
test_labels=array_th[per[train_size:],-1]  

model=find_fit(train_set,train_labels)      
alpha=model[0]
q=model[-1]


params=[]
 
params.append(model)

new_array=train_set

plt.figure(i)
z1=find_val(new_array,model)
#z2=my_func_g(new_array,*popt)
plt.scatter(new_array[:,5],train_labels/ps,marker='.',label='true')
plt.scatter(new_array[:,5],z1/ps,marker='.',label='pred')
#    plt.scatter(new_array[:,5],z2/ps,marker='.',label='curve_fit')

plt.xlabel('grain size')
plt.ylabel('execution time')
plt.xscale('log')
plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
plt.axvline(ps/th,color='gray',linestyle='dotted')  
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

i=i+1
    
th=2

for th in thr[1:]:
    array_th=array[array[:,2]==th]
    data_size=int(np.shape(array_th)[0])
    print(int(ps),data_size)
    
    train_size=data_size#int(np.ceil(data_size*0.6))
    tes_size=data_size-train_size
    
    per = np.random.permutation(data_size)
    train_set=array_th[per[0:train_size],:-1] 
    train_labels=array_th[per[0:train_size],-1]  
    test_set=array_th[per[train_size:],:-1]  
    test_labels=array_th[per[train_size:],-1]  

    model=find_fit(train_set,train_labels,params[0])        
    params.append(model)
    
    new_array=train_set

    plt.figure(i)
    z1=find_val(new_array,model)
#    z2=my_func_g(new_array,*popt)
    plt.scatter(new_array[:,5],train_labels/ps,marker='.',label='true')
    plt.scatter(new_array[:,5],z1/ps,marker='.',label='pred')
#    plt.scatter(new_array[:,5],z2/ps,marker='.',label='curve_fit')

    plt.xlabel('grain size')
    plt.ylabel('execution time')
    plt.xscale('log')
    plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
    plt.axvline(ps/th,color='gray',linestyle='dotted')  
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    i=i+1
    plt.figure(i)
    z1=find_val(new_array,model)
    z2=my_func_g(new_array,*popt)
    plt.scatter(new_array[:,5],ps/test_labels,marker='.',label='true')
    plt.scatter(new_array[:,5],ps/z1,marker='.',label='pred')
#    plt.scatter(new_array[:,5],z2/ps,marker='.',label='curve_fit')

    plt.xlabel('grain size')
    plt.ylabel('mflops')
    plt.xscale('log')
    plt.title('problem size:'+str(ps)+'  '+str(th)+' threads')
    plt.axvline(ps/th,color='gray',linestyle='dotted')  
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    i=i+1
    
    
labels=['alpha','gamma','d','h','q']
for i in range(5):    
    plt.figure(i+1)
    plt.scatter(thr,[p[i] for p in params],label=labels[i])    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.scatter(thr,[p[0]+p[-2] for p in params],label=labels[i])    

#############################################################
th=1
array_th=array[array[:,2]==th]
data_size=int(np.shape(array_th)[0])
print(int(ps),data_size)

train_size=int(np.ceil(data_size*0.6))
tes_size=data_size-train_size

per = np.random.permutation(data_size)
train_set=array_th[per[0:train_size],:-1] 
train_labels=array_th[per[0:train_size],-1]  
test_set=array_th[per[train_size:],:-1]  
test_labels=array_th[per[train_size:],-1]  

#remove outliers
np.abs(train_labels-ps)/1e6
indices=train_set[tra]

model=find_fit(train_set,train_labels)      
model

