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
from sklearn.metrics import r2_score


def create_dict(directories,to_csv=False):
    data_filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'
    thr=[]
    data_files=[]
    
    for directory in directories:
        [data_files.append(i) for i in glob.glob(directory+'/*.dat')]
    
    chunk_sizes=[]
    num_iterations=[]
    iter_lengths=[]
    nodes=[]
    
    if to_csv:
        f_csv=open(data_filename,'w')
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
#    marvin_dir='/home/shahrzad/repos/Blazemark/data/grain_size/marvin'
#    excludes=[marvin_dir+'/marvin_grain_size_1_1_2000_50000.dat',marvin_dir+'/marvin_grain_size_1_1_1000_100000.dat']
    excludes=[] 
    for filename in data_files:   
        if filename not in excludes:             
            f=open(filename, 'r')
                     
            result=f.readlines()
            if len(result)!=0:
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
tameshk_dir='/home/shahrzad/repos/Blazemark/data/grain_size/tameshk'
<<<<<<< HEAD

#results_dir='/home/shahrzad/repos/Blazemark/results/grain_size'
#create_dict([results_dir],1)

=======

#results_dir='/home/shahrzad/repos/Blazemark/results/grain_size'
#create_dict([results_dir],1)

>>>>>>> 358b65fe629cd2a890bca1682655efafafca5639
create_dict([marvin_dir,medusa_dir,tameshk_dir],1)

def my_func_g_5(ndata,alpha,gamma): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
    ts=ps
    return alpha*L+ts*(1+(gamma)*(M-1))*(w_c)/ps#+(1)*(d*ps)*np.exp(-((g-ps/N)/(k))**2)#+(1+(gamma)*(M-1))*(w_c)#+(1)*(1/(np.sqrt(2*np.pi)*(d)))*np.exp(-((g-dN)/(ps/N))**2)

titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'
#perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/1/all/'
perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'

dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)

nodes=dataframe['node'].drop_duplicates().values
nodes.sort()

problem_sizes=dataframe['problem_size'].drop_duplicates().values
problem_sizes.sort()
node='marvin'
i=1   
popt={}              
test_errors={}
r2_errors={}  
threads={}
for node in nodes:
    np.random.seed(0)                

    node_selected=dataframe['node']==node
    nt_selected=dataframe['num_tasks']>=1
    iter_selected=dataframe['iter_length']==1
    th_selected=dataframe['num_threads']>=1
    df_n_selected=dataframe[node_selected & nt_selected & iter_selected & th_selected][titles[1:]]
    
    thr=df_n_selected['num_threads'].drop_duplicates().values
    thr.sort()
    threads[node]=thr
    problem_sizes=df_n_selected['problem_size'].drop_duplicates().values
    problem_sizes.sort()



    array_all=df_n_selected.values
    array_all=array_all.astype(float)
#    base_ps=1e6
#    if node=='medusa':
#        base_ps=1e6
    test_errors[node]={}
    r2_errors[node]={}
    popt[node]={}
    base_pss=[1e4,1e5,1e6,1e7,1e8]
    base_ps=1e5
#    for base_ps in [base_ps for base_ps in base_pss if base_ps in problem_sizes]:
    test_errors[node][base_ps]={}
    r2_errors[node][base_ps]={}
    
    array_selected_ps=array_all[array_all[:,0]==base_ps]
    array=array_all[array_all[:,0]!=base_ps]
    
    per=np.array([])
    for th in thr:
        ind_th=np.where(array_all[:,2]==th)[0]
        data_size=int(1*np.shape(ind_th)[0])
        per = np.concatenate((ind_th[np.random.permutation(data_size)],per),axis=0)
    train_indices=per.astype(int)
                     
    train_size=np.shape(train_indices)[0]
    print(train_size)
    
    train_set=array_all[train_indices,:-1] 
    train_labels=array_all[train_indices,-1]  
    
#        all_indices=np.arange(np.shape(array_all)[0])
#        
#        test_indices=[ind for ind in all_indices if ind not in train_indices]
#        test_set=array_all[test_indices,:-1]  
#        test_labels=array_all[test_indices,-1]  
#        print(np.shape(test_set)[0])
    
    array_ps=array_selected_ps[:,:-1]
    labels_ps=array_selected_ps[:,-1]
    
    print(node,base_ps,np.shape(array_ps)[0])

    a_s=np.argsort(array_ps[:,5])
    for ir in range(np.shape(array_ps)[1]):
        array_ps[:,ir]=array_ps[a_s,ir]
    labels_ps=labels_ps[a_s]     
    
    param_bounds=([0,0],[np.inf,np.inf])

    popt_5, pcov=curve_fit(my_func_g_5,array_ps,labels_ps,method='trf',bounds=param_bounds)
    popt[node][base_ps]=popt_5
    test_errors[node][base_ps]={}
    r2_errors[node][base_ps]={}

    for ps in base_pss:#[ps for ps in problem_sizes]:
        array_ps=train_set[train_set[:,0]==ps]
        labels_ps=train_labels[train_set[:,0]==ps]
        
        a_s=np.argsort(array_ps[:,5])
        for ir in range(np.shape(array_ps)[1]):
            array_ps[:,ir]=array_ps[a_s,ir]
        labels_ps=labels_ps[a_s] 
        test_errors[node][base_ps][ps]={}
        r2_errors[node][base_ps][ps]={}
        lb=0.1
        ls=.1
#        for lb in [0.5,0.6,0.7,0.8]:
        for th in [8]:
            new_array=array_ps[array_ps[:,2]==th]
            new_labels=labels_ps[array_ps[:,2]==th]
            
            if np.shape(new_array[new_array[:,3]>0])[0]>30:
                plt.figure(i)
#                plt.axes([0, 0, 1.5, 1])

#                z_3=my_func_g_3(new_array,*popt_3)
                z_5=my_func_g_5(new_array,*popt_5)
                test_errors[node][base_ps][ps][th]=100*np.mean(np.abs(z_5-new_labels)/new_labels)
                r2_errors[node][base_ps][ps][th]=r2_score(new_labels,z_5)
                
#                opt=np.logical_and(new_array[:,5]>100, new_array[:,5]<2e6)
#                plt.scatter(new_array[:,5][opt],new_labels[opt],marker='.',label='true')
#                plt.scatter(new_array[:,5],new_labels,marker='.',label=str(int(th))+' threads')
                plt.scatter(new_array[:,5],new_labels,marker='.',label='true')

#                plt.scatter(new_array[:,5],z_5,marker='.',label='fitted')

                g1=np.ceil(np.sqrt(popt_5[0]*ps/(th*lb)))
                g3=np.ceil(np.sqrt(popt_5[0]*ps/(th*0.8)))
                g5=np.ceil(np.sqrt(popt_5[0]*ps/(th*0.4)))

                g2=np.floor(ps/(th*(1+np.ceil(1/ls))))
                g4=np.floor(ps/(th*(1+np.ceil(1/0.8))))
                g6=np.floor(ps/(th*(1+np.ceil(1/0.4))))

#                    plt.axvspan(1,70,color='red',alpha=0.1)  
                
#                plt.axvspan(70,2000,color='red',alpha=0.1)  
#                    plt.axvspan(1000,100000,color='blue',alpha=0.1)  

#                gg=np.linspace(g1,g2,1000)
#                for j in range(np.shape(gg)[0]):
#                    plt.axvline(gg[j],color='lavender')  
               
#                plt.axvline(g1)  
                plt.axvline(g3,color='green',alpha=0.2,label='$\lambda_b=0.8$') 
                plt.axvline(g5,color='green',alpha=0.6,label='$\lambda_b=0.4$')  
                plt.axvline(g1,color='green',alpha=1,label='$\lambda_b=0.1$')  


                plt.axvline(g2,color='red',alpha=0.2,label='$\lambda_s=0.1$')    
                plt.axvline(g6,color='red',alpha=0.6,label='$\lambda_s=0.4$')    
                plt.axvline(g4,color='red',alpha=1,label='$\lambda_s=0.8$')    
#                plt.axvspan(g1,g2,color='green',alpha=0.5)
#                plt.axvspan(g3,g2,color='green',alpha=0.4)
#                plt.axvspan(g2,g4,color='green',alpha=0.4)

#                plt.fill_between(new_array[:,5],where=np.logical_and(new_array[:,5]<=g2,new_array[:,5]>=g1),facecolor='green',alpha=.5)
                plt.xlabel('Grain size')
                plt.ylabel('Execution time')
                plt.xscale('log')
#                    print(lb,ls,g1,g2)
#                plt.title('problem size:'+str(int(ps))+'  '+str(int(th))+' threads')
#                plt.axvline(ps/(th),color='gray',linestyle='dotted')  
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#                    plt.savefig(perf_dir+'nows_new_rostam/'+str(int(ps))+'_'+str(int(th))+'_1_all.png',bbox_inches='tight')
#                plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_'+str(int(base_ps))+'.png',bbox_inches='tight')
#                    plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/'+node+'_'+str(int(ps))+'.png',bbox_inches='tight')

                i=i+1                    
                plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_range_10_50_80.png',bbox_inches='tight')

<<<<<<< HEAD
#                plt.scatter(new_array[:,5],z_5,marker='.',label='fitted')

                g1=np.ceil(np.sqrt(popt_5[0]*ps/(th*lb)))
                g3=np.ceil(np.sqrt(popt_5[0]*ps/(th*0.8)))
                g5=np.ceil(np.sqrt(popt_5[0]*ps/(th*0.4)))

                g2=np.floor(ps/(th*(1+np.ceil(1/ls))))
                g4=np.floor(ps/(th*(1+np.ceil(1/0.8))))
                g6=np.floor(ps/(th*(1+np.ceil(1/0.4))))

#                    plt.axvspan(1,70,color='red',alpha=0.1)  
                
#                plt.axvspan(70,2000,color='red',alpha=0.1)  
#                    plt.axvspan(1000,100000,color='blue',alpha=0.1)  

#                gg=np.linspace(g1,g2,1000)
#                for j in range(np.shape(gg)[0]):
#                    plt.axvline(gg[j],color='lavender')  
               
#                plt.axvline(g1)  
                plt.axvline(g3,color='green',alpha=0.2,label='$\lambda_b=0.8$') 
                plt.axvline(g5,color='green',alpha=0.6,label='$\lambda_b=0.4$')  
                plt.axvline(g1,color='green',alpha=1,label='$\lambda_b=0.1$')  


                plt.axvline(g2,color='red',alpha=0.2,label='$\lambda_s=0.1$')    
                plt.axvline(g6,color='red',alpha=0.6,label='$\lambda_s=0.4$')    
                plt.axvline(g4,color='red',alpha=1,label='$\lambda_s=0.8$')    
#                plt.axvspan(g1,g2,color='green',alpha=0.5)
#                plt.axvspan(g3,g2,color='green',alpha=0.4)
#                plt.axvspan(g2,g4,color='green',alpha=0.4)

#                plt.fill_between(new_array[:,5],where=np.logical_and(new_array[:,5]<=g2,new_array[:,5]>=g1),facecolor='green',alpha=.5)
                plt.xlabel('Grain size')
                plt.ylabel('Execution time')
                plt.xscale('log')
#                    print(lb,ls,g1,g2)
#                plt.title('problem size:'+str(int(ps))+'  '+str(int(th))+' threads')
#                plt.axvline(ps/(th),color='gray',linestyle='dotted')  
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#                    plt.savefig(perf_dir+'nows_new_rostam/'+str(int(ps))+'_'+str(int(th))+'_1_all.png',bbox_inches='tight')
#                plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_'+str(int(base_ps))+'.png',bbox_inches='tight')
#                    plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/'+node+'_'+str(int(ps))+'.png',bbox_inches='tight')

                i=i+1                    
                plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_range_10_50_80.png',bbox_inches='tight')

                plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_range_'+str(int(100*lb))+'_'+str(int(100*ls))+'.png',bbox_inches='tight')

for node in nodes:
    for base_ps in test_errors[node].keys():
        fig=plt.figure(i)
        ax = fig.add_subplot(111)
        width=0.25
        rects1 = ax.bar(threads[node],[test_errors[node][base_ps][base_ps][i] for i in threads[node]], width,label='Base problem size='+str(int(base_ps)))
        #rects2 = ax.bar(np.array([0,1,2])+width,[param_errors[i][1] for i in range(3)], width, color='seagreen',label='test')
        plt.xlabel('#cores')
        plt.ylabel('Relative Error(%)')
        plt.xticks(threads[node])
        #ax.set_xticklabels(parameters)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #    plt.savefig(perf_dir+'/fitted/'+node+'_relative_error_'+str(int(base_ps))+'.png',bbox_inches='tight')
    plt.figure(i)
#    plt.savefig(perf_dir+'/fitted/'+node+'_relative_error_compared.png',bbox_inches='tight')
    i=i+1

=======
>>>>>>> 358b65fe629cd2a890bca1682655efafafca5639
                plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_range_'+str(int(100*lb))+'_'+str(int(100*ls))+'.png',bbox_inches='tight')

for node in nodes:
    for base_ps in test_errors[node].keys():
        fig=plt.figure(i)
        ax = fig.add_subplot(111)
        width=0.25
        rects1 = ax.bar(threads[node],[test_errors[node][base_ps][base_ps][i] for i in threads[node]], width,label='Base problem size='+str(int(base_ps)))
        #rects2 = ax.bar(np.array([0,1,2])+width,[param_errors[i][1] for i in range(3)], width, color='seagreen',label='test')
        plt.xlabel('#cores')
        plt.ylabel('Relative Error(%)')
        plt.xticks(threads[node])
        #ax.set_xticklabels(parameters)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #    plt.savefig(perf_dir+'/fitted/'+node+'_relative_error_'+str(int(base_ps))+'.png',bbox_inches='tight')
    plt.figure(i)
#    plt.savefig(perf_dir+'/fitted/'+node+'_relative_error_compared.png',bbox_inches='tight')
    i=i+1

for node in nodes:
    k=0
    for base_ps in test_errors[node].keys():
        fig=plt.figure(i)
        ax = fig.add_subplot(111)
        width=0.15
        rects1 = ax.bar(threads[node]-.25+width*k, [r2_errors[node][base_ps][base_ps][i] for i in threads[node]], width,label='problem size='+str(int(base_ps)))
        k=k+1
        ax.set_xlabel('#cores')
        plt.ylabel('Relative Error(%)')
        plt.xticks(threads[node])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig=plt.figure(i)
    plt.savefig(perf_dir+'/fitted/'+node+'_r2_error_compared_self.png',bbox_inches='tight')
    i=i+1    
    
<<<<<<< HEAD
=======
    
>>>>>>> 358b65fe629cd2a890bca1682655efafafca5639
    

p_th={}

for node in nodes:
    p_th[node]={}
    for base_ps in test_errors[node].keys():
        p_th[node][base_ps]={}        
        for th in threads[node]:
            if node=='tameshk':
                p_th[node][base_ps][th]=np.mean([r2_errors[node][base_ps][ps][th] for ps in r2_errors[node][base_ps].keys() if th in test_errors[node][base_ps][ps].keys() and ps!=base_ps and ps!=1e8])
            else:
                p_th[node][base_ps][th]=np.mean([r2_errors[node][base_ps][ps][th] for ps in r2_errors[node][base_ps].keys() if th in test_errors[node][base_ps][ps].keys() and ps!=base_ps])

    k=0
    width = 0.15   
    fig=plt.figure(i)
    ax = fig.add_subplot(111)
    for base_ps in test_errors[node].keys():    
        rects1 = ax.bar(threads[node]-.25+width*k, [p_th[node][base_ps][th] for th in threads[node]], width,label='problem size='+str(int(base_ps)))
        k=k+1
        ax.set_xlabel('#cores')
        plt.ylabel('Relative Error(%)')
        plt.xticks(threads[node])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1
    plt.savefig(perf_dir+'/fitted/'+node+'_r2_error_compared_rest.png',bbox_inches='tight')
        
fig=plt.figure()
ax = fig.add_subplot(111)
width=0.25
rects1 = ax.bar(thr,[p_th[th] for th in thr], width, color='royalblue',label='training')
#rects2 = ax.bar(np.array([0,1,2])+width,[param_errors[i][1] for i in range(3)], width, color='seagreen',label='test')
plt.xlabel('#cores')
plt.ylabel('Relative Error(%)')
plt.xticks(thr)
#ax.set_xticklabels(parameters)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(perf_dir+'/fitted/'+node+'_relative_error_all_'+str(int(base_ps))+'.png',bbox_inches='tight')

r_th={}
for th in thr:
    if node=='tameshk':
        r_th[th]=np.mean([r2_errors[node][base_ps][ps][th] for ps in r2_errors[node][base_ps].keys() if th in r2_errors[node][base_ps][ps].keys() and ps!=base_ps and ps!=1e8])
    else:
        r_th[th]=np.mean([r2_errors[node][base_ps][ps][th] for ps in r2_errors[node][base_ps].keys() if th in r2_errors[node][base_ps][ps].keys() and ps!=base_ps])

    
fig=plt.figure()
ax = fig.add_subplot(111)
width=0.25
rects1 = ax.bar(thr,[r_th[th] for th in thr], width, color='royalblue',label='training')
#rects2 = ax.bar(np.array([0,1,2])+width,[param_errors[i][1] for i in range(3)], width, color='seagreen',label='test')
plt.xlabel('#cores')
plt.ylabel('$R^2\:{Score}$')
plt.xticks(thr)
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
#ax.set_xticklabels(parameters)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(perf_dir+'/fitted/'+node+'_r2_error_all_'+str(int(base_ps))+'.png',bbox_inches='tight')

def grain_dict(array,avg=False):
    g_dict={}
    
    g=array[:,3]
    p=array[:,-1]
    t=array[:,2]
    nt=array[:,-2]
    
    for i in range(len(g)):
        if g[i] not in g_dict.keys():
            g_dict[g[i]]={}
        if t[i] not in g_dict[g[i]].keys():
            g_dict[g[i]][t[i]]=[[],[]]
        g_dict[g[i]][t[i]][0].append(p[i])
        g_dict[g[i]][t[i]][1].append(nt[i])

    if avg:
        for gd in g_dict.keys():
            for td in g_dict[gd].keys():
                g_dict[gd][td][0]=sum(g_dict[gd][td][0])/len(g_dict[gd][td][0])
                g_dict[gd][td][1]=np.ceil(sum(g_dict[gd][td][1])/len(g_dict[gd][td][1]))
    return g_dict

def create_dict_refernce(directory):
    thr=[]
    nodes=[]
    data_files=glob.glob(directory+'/*.dat')
    benchmark=''
    benchmarks=[]
    
    for filename in data_files:
        try:
            (node, benchmark, th) = filename.split('/')[-1].replace('.dat','').split('-')   
        except:
            (node, benchmark, th, ref) = filename.split('/')[-1].replace('.dat','').split('-')  
        if benchmark not in benchmarks:
                benchmarks.append(benchmark)                
        if int(th) not in thr:
            thr.append(int(th))
        if node not in nodes:
            nodes.append(node)        
                  
    thr.sort()
    benchmarks.sort()      
    nodes.sort()      
    
    d_all={}   
    for node in nodes:
        d_all[node]={}
        for benchmark in benchmarks:  
            d_all[node][benchmark]={}
            for th in thr:
                d_all[node][benchmark][th]={}     
                                            
    data_files.sort()        
    for filename in data_files:                
        stop=False
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        try:
            (node, benchmark, th) = filename.split('/')[-1].replace('.dat','').split('-')   
        except:
            (node, benchmark, th, ref) = filename.split('/')[-1].replace('.dat','').split('-')          
        th = int(th)
        size=[]
        mflops=[]    
        for r in result:        
            if "N=" in r or '/' in r:
                stop=True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))
            
        d_all[node][benchmark][th]['size']=size
        d_all[node][benchmark][th]['mflops']=mflops
 
            
    return d_all                           

hpx_dir_ref='/home/shahrzad/repos/Blazemark/data/matrix/09-15-2019/reference-chunk_size_fixed/'         
d_hpx_ref=create_dict_refernce(hpx_dir_ref)  
        
b_filename='/home/shahrzad/repos/Blazemark/data/data_perf_all.csv'
titles=['runtime','node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','work_per_core','w1','w2','w3','w4','w5','w6','w7','w8','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','set_associativity','datatype','cost','simd_size','execution_time','num_tasks','mflops','include']

dataframe = pandas.read_csv(b_filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[3:]:
    dataframe[col] = dataframe[col].astype(float)
  

runtime='hpx'
benchmark='dmatdmatadd'

nodes=dataframe['node'].drop_duplicates().values
nodes.sort()
g_params={}
threads={}
included=dataframe['include']==1
node_selected=dataframe['node']==node
df_n_selected=dataframe[node_selected & included]

g_params[node]={}

benchmarks=df_n_selected['benchmark'].drop_duplicates().values
benchmarks.sort()
threads[node]={}
g_params[node][benchmark]={}
benchmark_selected=dataframe['benchmark']==benchmark
rt_selected=dataframe['runtime']==runtime
num_threads_selected=dataframe['num_threads']<=8
df_nb_selected=df_n_selected[benchmark_selected & num_threads_selected & rt_selected]         
block_selected_r=df_nb_selected['block_size_row']==4
block_selected_c=df_nb_selected['block_size_col']!=64
df_nb_selected=df_nb_selected[ block_selected_r | block_selected_c]
df_nb_selected=df_nb_selected[ block_selected_r ]#| block_selected_c]

matrix_sizes=df_nb_selected['matrix_size'].drop_duplicates().values
matrix_sizes.sort()
thr=df_nb_selected['num_threads'].drop_duplicates().values
thr.sort()
threads[node][benchmark]=thr

i=1
test_errors={}
r2_errors={}
lb=0.01
ls=0.5
chunk_sizes={}
for m in matrix_sizes:#[690,912,1825,3193,4222,4855,6420]:#matrix_sizes:
    chunk_sizes[m]={}
    chunk_sizes[m][str(lb)+'_'+str(ls)]={}
    test_errors[m]={}
    r2_errors[m]={}
    simdsize=4.
    if node=='medusa':
        simdsize=8.

    aligned_m=m
    if m%simdsize!=0:
        aligned_m=m+simdsize-m%simdsize
    if benchmark=='dmatdmatadd':                            
        mflop=(aligned_m)*m                           
    elif benchmark=='dmatdmatdmatadd':
        mflop=2*(aligned_m)*m
    else:
        mflop=2*(aligned_m)**3        
    
    m_selected=df_nb_selected['matrix_size']==m
    features=['chunk_size','num_blocks','num_threads','grain_size','block_size_row','block_size_col','work_per_core','num_tasks','execution_time']
    df_selected=df_nb_selected[m_selected][features]

    array_b=df_selected.values
    if np.shape(array_b)[0]!=0:
        array_b=array_b.astype(float)
      
        a_s=np.argsort(array_b[:,0])
        
        array_b=array_b[a_s]
        g_params[node][benchmark]=grain_dict(array_b,1)
        
       
        for th in [8]:#range(1,9):  
            b='4-256'
            b_r=int(b.split('-')[0])
            b_c=int(b.split('-')[1])
            if b_r>m:
                b_r=m
            if b_c>m:
                b_c=m
            if b_c%simdsize!=0:
                b_c=b_c+simdsize-b_c%simdsize
            
            equalshare1=math.ceil(m/b_r)
            equalshare2=math.ceil(m/b_c)  
            num_blocks=equalshare1*equalshare2
            aligned_m=m
            if m%simdsize!=0:
                aligned_m=m+simdsize-m%simdsize
                       
            mflop_c=0
            if benchmark=='dmatdmatadd':                            
                mflop_c=b_r*b_c                            
            elif benchmark=='dmatdmatdmatadd':
                mflop_c=b_r*b_c*2
            else:
                mflop_c=b_r*b_c*(2*m)
                
            num_elements=[mflop_c]*num_blocks
            if aligned_m%b_c!=0:
                for j in range(1,equalshare1+1):
                    if benchmark=='dmatdmatadd':                            
                        num_elements[j*equalshare2-1]=(aligned_m%b_c)*b_r                            
                    elif benchmark=='dmatdmatdmatadd':
                        num_elements[j*equalshare2-1]=(aligned_m%b_c)*b_r*2
                    else:
                        num_elements[j*equalshare2-1]=(aligned_m%b_c)*b_r*(2*m)
                
            if m%b_r!=0:
                for j in range(1,equalshare2+1):
                    if benchmark=='dmatdmatadd':                            
                        num_elements[(equalshare1-1)*equalshare2+j-1]=(m%b_r)*b_c                            
                    elif benchmark=='dmatdmatdmatadd':
                        num_elements[(equalshare1-1)*equalshare2+j-1]=(m%b_r)*b_c*2
                    else:
                        num_elements[(equalshare1-1)*equalshare2+j-1]=(m%b_r)*b_c*(2*m)
                                                                                       
            if aligned_m%b_c!=0 and m%b_r!=0:
                if benchmark=='dmatdmatadd':                            
                    num_elements[-1]=(m%b_r)*(aligned_m%b_c)                   
                elif benchmark=='dmatdmatdmatadd':
                    num_elements[-1]=(m%b_r)*(aligned_m%b_c)*2
                else:
                    num_elements[-1]=(m%b_r)*(aligned_m%b_c)*(2*m)
                    
            new_array=array_b[np.logical_and(array_b[:,2]==th,array_b[:,1]==num_blocks)][:,:-1]
            new_labels=array_b[np.logical_and(array_b[:,2]==th,array_b[:,1]==num_blocks)][:,-1]

#            new_array=array_b[array_b[:,2]==th][:,:-1]
#            new_labels=array_b[array_b[:,2]==th][:,-1]
        
            all_chunk_sizes=np.unique(new_array[:,0])
            all_chunk_sizes.sort()
    
            ts=g_params[node][benchmark][mflop][1][0]
            def my_func_g_b(ndata,alpha,gamma): 
                N=ndata[:,2]
                n_t=ndata[:,-1]
                M=np.minimum(n_t,N) 
                L=np.ceil(n_t/(M))
                w_c=ndata[:,-2]
                ps=mflop
                return alpha*L+ts*(1+(gamma)*(M-1))*(w_c)/ps#+(1)*(d*ps)*np.exp(-((g-ps/N)/(k))**2)#+(1+(gamma)*(M-1))*(w_c)#+(1)*(1/(np.sqrt(2*np.pi)*(d)))*np.exp(-((g-dN)/(ps/N))**2)
            
<<<<<<< HEAD
   
    for th in range(1,9):          
        new_array=array_b[array_b[:,2]==th][:,:-1]
        new_labels=array_b[array_b[:,2]==th][:,-1]


        ts=g_params[node][benchmark][mflop][1][0]
        def my_func_g_b(ndata,alpha,gamma): 
            N=ndata[:,2]
            n_t=ndata[:,-1]
            M=np.minimum(n_t,N) 
            L=np.ceil(n_t/(M))
            w_c=ndata[:,-2]
=======
>>>>>>> 358b65fe629cd2a890bca1682655efafafca5639
            ps=mflop
            zb=my_func_g_b(new_array,*popt_5)
    
            
            test_errors[m][th]=np.mean(np.abs(zb-new_labels))#/new_labels)
            r2_errors[m][th]=r2_score(new_labels,zb)
            
    #            test_errors[m][th]=100*np.mean(np.abs(new_labels-zb-(np.median(new_labels)-np.median(zb)))/new_labels)
    #            r2_errors[m][th]=r2_score(new_labels-(np.median(new_labels)-np.median(zb)),zb)
    #        
    #        for lb in [0.001,0.005,0.01,0.05,0.1]:
    #            for ls in [0.1,0.2,0.5]:
#            lb=0.01
#            ls=.5
            g1=np.ceil(np.sqrt(popt_5[0]*ps/(th*lb)))
            g2=np.floor(ps/(th*(1+np.ceil(1/ls))))
            plt.figure(i)
    ##        plt.axes([0, 0, 2, 1])
#            plt.scatter(new_array[:,3],new_labels,color='blue',label='true',marker='.')
#            plt.scatter(new_array[:,3],zb,label='pred',marker='.',color='red')
##            plt.scatter(new_array[:,3][new_array[:,4]==8],new_labels[new_array[:,4]==8],color='green',label='true',marker='.')
#            plt.axvline(mflop/th,color='gray',linestyle='dashed')
#
#            plt.axvspan(g1,g2,color='green',alpha=0.5)
    ##                plt.scatter(new_array[:,3],new_labels-zb,label='pred',marker='.',color='red')
    #
    ##        plt.grid(True,'both')
#            plt.xscale('log')
#            plt.xlabel('Grain size')
#            plt.ylabel('Execution time')
#    ##        plt.title('test set  matrix size:'+str(int(m))+'  '+str(int(th))+' threads')
#    #
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#            plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+node+'_pred_'+str(int(m))+'_'+str(int(th))+'.png',bbox_inches='tight')
#            i=i+1
#            plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/range/'+node+'_pred_'+str(int(m))+'_'+str(int(th))+'_'+str(int(1000*lb))+'_'+str(int(1000*ls))+'.png',bbox_inches='tight')

#            plt.savefig(perf_dir+'/blazemark/range/'+node+'_pred_'+str(int(m))+'_'+str(int(th))+'_'+str(int(1000*lb))+'_'+str(int(1000*ls))+'.png',bbox_inches='tight')
            
            chunk_sizes[m][str(lb)+'_'+str(ls)][th]=[1,mflop]    
            first=True      
            last=True
            range_c=[1,mflop]   
            plt.figure(i)  
            all_values=[new_labels[np.logical_and(new_array[:,0]==c,new_array[:,1]==num_blocks)][0] for c in all_chunk_sizes]
            plt.scatter(all_chunk_sizes,all_values,color='royalblue')
    
            for c in range(1,num_blocks):  
                grain_size=sum(num_elements[0:c])
    #            print(grain_size,g1,g2)
                if grain_size>=g1:
                    if first:                    
                        range_c[0]=c
                        first=False
                    else:
                        if grain_size>g2:
                            range_c[1]=c-1    
                            break
                        else:
                            if c in all_chunk_sizes:
                                plt.figure(i)
    
                                plt.scatter(c,new_labels[np.logical_and(new_array[:,0]==c,new_array[:,1]==num_blocks)][0],color='red')
                                plt.xlabel('Chunk size')
                                plt.ylabel('Execution time')       
                                plt.xscale('log')
            plt.figure(i)
            plt.axvspan(range_c[0],range_c[1],color='red',alpha=0.25)
            if m in d_hpx_ref[node][benchmark][th]['size']:
                k=d_hpx_ref[node][benchmark][th]['size'].index(m)  
                v=d_hpx_ref[node][benchmark][th]['mflops'][k]                              
                plt.axhline(mflop/v,color='green')
            plt.xlabel('Chunk size')
            plt.ylabel('Execution time')       
            plt.xscale('log')
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+node+'_compared_'+str(int(m))+'_'+str(int(th))+'_'+str(int(1000*lb))+'_'+str(int(1000*ls))+'_'+str(range_c[0])+'_'+str(range_c[1])+'.png',bbox_inches='tight')

#            plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+node+'_pred_'+str(int(m))+'_'+str(int(th))+'_'+str(int(1000*lb))+'_'+str(int(1000*ls))+'_'+str(range_c[0])+'_'+str(range_c[1])+'.png',bbox_inches='tight')
#            plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+benchmark+'/'+node+'_pred_'+str(int(m))+'_'+str(int(th))+'_'+str(int(1000*lb))+'_'+str(int(1000*ls))+'_'+str(range_c[0])+'_'+str(range_c[1])+'.png',bbox_inches='tight')
    
    
            chunk_sizes[m][str(lb)+'_'+str(ls)][th]=range_c
            i=i+1
            
                
p_th={}
for th in thr:
    p_th[th]=np.mean([test_errors[m][th] for m in test_errors.keys() if th in test_errors[m].keys() and m<953])
    
fig=plt.figure()
ax = fig.add_subplot(111)
width=0.25
rects1 = ax.bar(np.arange(1,9),[p_th[th] for th in thr], width, color='royalblue',label='training')
#rects2 = ax.bar(np.array([0,1,2])+width,[param_errors[i][1] for i in range(3)], width, color='seagreen',label='test')
plt.xlabel('#cores')
plt.ylabel('Relative Error(%)')
plt.xticks(np.arange(1,9))
#ax.set_xticklabels(parameters)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.savefig(perf_dir+'/blazemark/'+node+'_relative_error_all.png',bbox_inches='tight')
<<<<<<< HEAD

plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+node+'_relative_error_less_953.png',bbox_inches='tight')

r_th={}
for th in thr:
    r_th[th]=np.mean([r2_errors[m][th] for m in test_errors.keys() if th in test_errors[m].keys() and m<953])
        #    ic=L*N%n_t
        #    g=ndata[:,5]
        #    k=ps/(N*(N-1))
        #    return alpha*L+(1)*(w_c)+(w_c-ps%g)/(np.ceil((N-2)/(N-n_t))-1)           
        #    return alpha*L+(1)*w_c+(ic)*np.ceil((ps%g)/(N-ic-1+0.000001))+(ic+np.ceil(ps%g/g))*np.ceil((g-ps%g)/(N-ic+0.000001))
        #    return alpha*L+(1)*w_c+ic*(ps%g)/((N-ic))+(ic!=N-1)*(g-ps%g)*(ic+np.ceil((ps%g)/g))/((N-ic-np.ceil((ps%g)/g)+0.00001))
            return alpha*L+ts*(1+(gamma)*(M-1))*(w_c)/ps#+(1)*(d*ps)*np.exp(-((g-ps/N)/(k))**2)#+(1+(gamma)*(M-1))*(w_c)#+(1)*(1/(np.sqrt(2*np.pi)*(d)))*np.exp(-((g-dN)/(ps/N))**2)
        ps=mflop
        zb=my_func_g_b(new_array,*popt_5)

        
        array_b=array_b[a_s]
        g_params[node][benchmark]=grain_dict(array_b,1)
        
       
        for th in [8]:#range(1,9):  
            b='4-256'
            b_r=int(b.split('-')[0])
            b_c=int(b.split('-')[1])
            if b_r>m:
                b_r=m
            if b_c>m:
                b_c=m
            if b_c%simdsize!=0:
                b_c=b_c+simdsize-b_c%simdsize
            
            equalshare1=math.ceil(m/b_r)
            equalshare2=math.ceil(m/b_c)  
            num_blocks=equalshare1*equalshare2
            aligned_m=m
            if m%simdsize!=0:
                aligned_m=m+simdsize-m%simdsize
                       
            mflop_c=0
            if benchmark=='dmatdmatadd':                            
                mflop_c=b_r*b_c                            
            elif benchmark=='dmatdmatdmatadd':
                mflop_c=b_r*b_c*2
            else:
                mflop_c=b_r*b_c*(2*m)
                
            num_elements=[mflop_c]*num_blocks
            if aligned_m%b_c!=0:
                for j in range(1,equalshare1+1):
                    if benchmark=='dmatdmatadd':                            
                        num_elements[j*equalshare2-1]=(aligned_m%b_c)*b_r                            
                    elif benchmark=='dmatdmatdmatadd':
                        num_elements[j*equalshare2-1]=(aligned_m%b_c)*b_r*2
                    else:
                        num_elements[j*equalshare2-1]=(aligned_m%b_c)*b_r*(2*m)
                
            if m%b_r!=0:
                for j in range(1,equalshare2+1):
                    if benchmark=='dmatdmatadd':                            
                        num_elements[(equalshare1-1)*equalshare2+j-1]=(m%b_r)*b_c                            
                    elif benchmark=='dmatdmatdmatadd':
                        num_elements[(equalshare1-1)*equalshare2+j-1]=(m%b_r)*b_c*2
                    else:
                        num_elements[(equalshare1-1)*equalshare2+j-1]=(m%b_r)*b_c*(2*m)
                                                                                       
            if aligned_m%b_c!=0 and m%b_r!=0:
                if benchmark=='dmatdmatadd':                            
                    num_elements[-1]=(m%b_r)*(aligned_m%b_c)                   
                elif benchmark=='dmatdmatdmatadd':
                    num_elements[-1]=(m%b_r)*(aligned_m%b_c)*2
                else:
                    num_elements[-1]=(m%b_r)*(aligned_m%b_c)*(2*m)
                    
            new_array=array_b[np.logical_and(array_b[:,2]==th,array_b[:,1]==num_blocks)][:,:-1]
            new_labels=array_b[np.logical_and(array_b[:,2]==th,array_b[:,1]==num_blocks)][:,-1]

#            new_array=array_b[array_b[:,2]==th][:,:-1]
#            new_labels=array_b[array_b[:,2]==th][:,-1]
        
            all_chunk_sizes=np.unique(new_array[:,0])
            all_chunk_sizes.sort()
    
            ts=g_params[node][benchmark][mflop][1][0]
            def my_func_g_b(ndata,alpha,gamma): 
                N=ndata[:,2]
                n_t=ndata[:,-1]
                M=np.minimum(n_t,N) 
                L=np.ceil(n_t/(M))
                w_c=ndata[:,-2]
                ps=mflop
                return alpha*L+ts*(1+(gamma)*(M-1))*(w_c)/ps#+(1)*(d*ps)*np.exp(-((g-ps/N)/(k))**2)#+(1+(gamma)*(M-1))*(w_c)#+(1)*(1/(np.sqrt(2*np.pi)*(d)))*np.exp(-((g-dN)/(ps/N))**2)
            
            ps=mflop
            zb=my_func_g_b(new_array,*popt_5)
    
            
            test_errors[m][th]=np.mean(np.abs(zb-new_labels))#/new_labels)
            r2_errors[m][th]=r2_score(new_labels,zb)
            
    #            test_errors[m][th]=100*np.mean(np.abs(new_labels-zb-(np.median(new_labels)-np.median(zb)))/new_labels)
    #            r2_errors[m][th]=r2_score(new_labels-(np.median(new_labels)-np.median(zb)),zb)
    #        
    #        for lb in [0.001,0.005,0.01,0.05,0.1]:
    #            for ls in [0.1,0.2,0.5]:
#            lb=0.01
#            ls=.5
            g1=np.ceil(np.sqrt(popt_5[0]*ps/(th*lb)))
            g2=np.floor(ps/(th*(1+np.ceil(1/ls))))
            plt.figure(i)
    ##        plt.axes([0, 0, 2, 1])
#            plt.scatter(new_array[:,3],new_labels,color='blue',label='true',marker='.')
#            plt.scatter(new_array[:,3],zb,label='pred',marker='.',color='red')
##            plt.scatter(new_array[:,3][new_array[:,4]==8],new_labels[new_array[:,4]==8],color='green',label='true',marker='.')
#            plt.axvline(mflop/th,color='gray',linestyle='dashed')
#
#            plt.axvspan(g1,g2,color='green',alpha=0.5)
    ##                plt.scatter(new_array[:,3],new_labels-zb,label='pred',marker='.',color='red')
    #
    ##        plt.grid(True,'both')
#            plt.xscale('log')
#            plt.xlabel('Grain size')
#            plt.ylabel('Execution time')
#    ##        plt.title('test set  matrix size:'+str(int(m))+'  '+str(int(th))+' threads')
#    #
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#            plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+node+'_pred_'+str(int(m))+'_'+str(int(th))+'.png',bbox_inches='tight')
#            i=i+1
#            plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/range/'+node+'_pred_'+str(int(m))+'_'+str(int(th))+'_'+str(int(1000*lb))+'_'+str(int(1000*ls))+'.png',bbox_inches='tight')

#            plt.savefig(perf_dir+'/blazemark/range/'+node+'_pred_'+str(int(m))+'_'+str(int(th))+'_'+str(int(1000*lb))+'_'+str(int(1000*ls))+'.png',bbox_inches='tight')
            
            chunk_sizes[m][str(lb)+'_'+str(ls)][th]=[1,mflop]    
            first=True      
            last=True
            range_c=[1,mflop]   
            plt.figure(i)  
            all_values=[new_labels[np.logical_and(new_array[:,0]==c,new_array[:,1]==num_blocks)][0] for c in all_chunk_sizes]
            plt.scatter(all_chunk_sizes,all_values,color='royalblue')
    
            for c in range(1,num_blocks):  
                grain_size=sum(num_elements[0:c])
    #            print(grain_size,g1,g2)
                if grain_size>=g1:
                    if first:                    
                        range_c[0]=c
                        first=False
                    else:
                        if grain_size>g2:
                            range_c[1]=c-1    
                            break
                        else:
                            if c in all_chunk_sizes:
                                plt.figure(i)
    
                                plt.scatter(c,new_labels[np.logical_and(new_array[:,0]==c,new_array[:,1]==num_blocks)][0],color='red')
                                plt.xlabel('Chunk size')
                                plt.ylabel('Execution time')       
                                plt.xscale('log')
            plt.figure(i)
            plt.axvspan(range_c[0],range_c[1],color='red',alpha=0.25)
            if m in d_hpx_ref[node][benchmark][th]['size']:
                k=d_hpx_ref[node][benchmark][th]['size'].index(m)  
                v=d_hpx_ref[node][benchmark][th]['mflops'][k]                              
                plt.axhline(mflop/v,color='green')
            plt.xlabel('Chunk size')
            plt.ylabel('Execution time')       
            plt.xscale('log')
#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+node+'_compared_'+str(int(m))+'_'+str(int(th))+'_'+str(int(1000*lb))+'_'+str(int(1000*ls))+'_'+str(range_c[0])+'_'+str(range_c[1])+'.png',bbox_inches='tight')

#            plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+node+'_pred_'+str(int(m))+'_'+str(int(th))+'_'+str(int(1000*lb))+'_'+str(int(1000*ls))+'_'+str(range_c[0])+'_'+str(range_c[1])+'.png',bbox_inches='tight')
#            plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+benchmark+'/'+node+'_pred_'+str(int(m))+'_'+str(int(th))+'_'+str(int(1000*lb))+'_'+str(int(1000*ls))+'_'+str(range_c[0])+'_'+str(range_c[1])+'.png',bbox_inches='tight')
    
    
            chunk_sizes[m][str(lb)+'_'+str(ls)][th]=range_c
            i=i+1
            
                
p_th={}
for th in thr:
    p_th[th]=np.mean([test_errors[m][th] for m in test_errors.keys() if th in test_errors[m].keys() and m<953])
    
fig=plt.figure()
ax = fig.add_subplot(111)
width=0.25
rects1 = ax.bar(np.arange(1,9),[p_th[th] for th in thr], width, color='royalblue',label='training')
#rects2 = ax.bar(np.array([0,1,2])+width,[param_errors[i][1] for i in range(3)], width, color='seagreen',label='test')
plt.xlabel('#cores')
plt.ylabel('Relative Error(%)')
plt.xticks(np.arange(1,9))
#ax.set_xticklabels(parameters)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.savefig(perf_dir+'/blazemark/'+node+'_relative_error_all.png',bbox_inches='tight')

plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+node+'_relative_error_less_953.png',bbox_inches='tight')


plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+node+'_relative_error_less_953.png',bbox_inches='tight')

r_th={}
for th in thr:
    r_th[th]=np.mean([r2_errors[m][th] for m in test_errors.keys() if th in test_errors[m].keys() and m<953])
    
fig=plt.figure()
ax = fig.add_subplot(111)
width=0.25
rects1 = ax.bar(np.arange(1,9),[r_th[th] for th in thr], width, color='royalblue',label='training')
#rects2 = ax.bar(np.array([0,1,2])+width,[param_errors[i][1] for i in range(3)], width, color='seagreen',label='test')
plt.xlabel('#cores')
plt.ylabel('$R^2\:{Score}$')
plt.xticks(np.arange(1,9))
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
#ax.set_xticklabels(parameters)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+node+'_r2_error_all.png',bbox_inches='tight')
plt.savefig(perf_dir+'/blazemark/'+str(int(base_ps))+'/'+node+'_r2_error_less_953.png',bbox_inches='tight')
plt.savefig(perf_dir+'/blazemark/'+node+'_r2_error_all.png',bbox_inches='tight')
