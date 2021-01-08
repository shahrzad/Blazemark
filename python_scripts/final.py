#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:10:26 2020

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
matplotlib.rcParams.update({'font.size': 30})

filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf.csv'
marvin_dir='/home/shahrzad/repos/Blazemark/data/final/grain_size/marvin/general/'
medusa_dir='/home/shahrzad/repos/Blazemark/data/final/grain_size/medusa/general/'

gf.create_dict([marvin_dir,medusa_dir],data_filename=filename)

def my_func(ndata,alpha,gamma): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
    ts=ps
    return alpha*L+ts*(1+(gamma)*(M-1))*(w_c)/ps#+(1)*(d*ps)*np.exp(-((g-ps/N)/(k))**2)#+(1+(gamma)*(M-1))*(w_c)#+(1)*(1/(np.sqrt(2*np.pi)*(d)))*np.exp(-((g-dN)/(ps/N))**2)

titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']

perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general/thesis'

dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)

nodes=dataframe['node'].drop_duplicates().values
nodes.sort()


#node='marvin'
i=1   
popt={}              
test_errors={}
r2_errors={}  
node_parameters={}
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
    problem_sizes=df_n_selected['problem_size'].drop_duplicates().values
    problem_sizes.sort()

    threads[node]=thr


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
#    for base_ps in base_pss: 
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
    
    
    array_ps=array_selected_ps[:,:-1]
    labels_ps=array_selected_ps[:,-1]
    
    print(node,base_ps,np.shape(array_ps)[0])

    a_s=np.argsort(array_ps[:,5])
    for ir in range(np.shape(array_ps)[1]):
        array_ps[:,ir]=array_ps[a_s,ir]
    labels_ps=labels_ps[a_s]     
    
    param_bounds=([0,0],[np.inf,np.inf])

    popt_5, pcov=curve_fit(my_func,array_ps,labels_ps,method='trf',bounds=param_bounds)
    node_parameters[node]=popt_5
    popt[node][base_ps]=popt_5
    test_errors[node][base_ps]={}
    r2_errors[node][base_ps]={}

#    ps=1e5
    for ps in [1e5, 1e8]:#base_pss:#[ps for ps in problem_sizes]:
        print(base_ps, ps)
        array_ps=train_set[train_set[:,0]==ps]
        labels_ps=train_labels[train_set[:,0]==ps]
        
        a_s=np.argsort(array_ps[:,5])
        for ir in range(np.shape(array_ps)[1]):
            array_ps[:,ir]=array_ps[a_s,ir]
        labels_ps=labels_ps[a_s] 
        test_errors[node][base_ps][ps]={}
        r2_errors[node][base_ps][ps]={}
        
#        for th in thr:
#            new_array=array_ps[array_ps[:,2]==th]
#            new_labels=labels_ps[array_ps[:,2]==th]
#            z_5=my_func(new_array,*popt_5)
#            test_errors[node][base_ps][ps][th]=100*np.mean(np.abs(z_5-new_labels)/new_labels)
#            r2_errors[node][base_ps][ps][th]=r2_score(new_labels,z_5)
#        
#        lb=0.5
#        ls=.1
#        for lb in [0.5,0.6,0.7,0.8]:

#plot all threads
        for th in thr:
            plt.figure(i)
            new_array=array_ps[array_ps[:,2]==th]
            new_labels=labels_ps[array_ps[:,2]==th]
#                if th==1:
#                    label='1 thread'
#                else:
#                    label=str(int(th))+' threads'
#                plt.axes([0, 0, 1.5, 1.5])
#
#                plt.scatter(new_array[:,5],new_labels,marker='.',label=label)
#                plt.xlabel('Grain size')
#                plt.ylabel('Execution Time($\mu{sec}$)')
#                plt.xscale('log')
#                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
#            plt.savefig(perf_dir+'/'+node+'_'+str(int(ps))+'_all.png',bbox_inches='tight')

#                if np.shape(new_array[new_array[:,3]>0])[0]>30:  
            z_5=my_func(new_array,*popt_5)
            test_errors[node][base_ps][ps][th]=100*np.mean(np.abs(z_5-new_labels)/new_labels)
            r2_errors[node][base_ps][ps][th]=r2_score(new_labels,z_5)
            plt.axes([0, 0, 1.5, 1.5])

            plt.scatter(new_array[:,5],new_labels,label='true')
#            plt.axvspan(70,1000,color='red',alpha=0.2)
  

            plt.scatter(new_array[:,5],z_5,label='prediction')
            plt.xlabel('Grain size')
            plt.ylabel('Execution Time($\mu{sec}$)')
            plt.xscale('log')
            plt.legend(bbox_to_anchor=(0.08, 0.98), loc=2, borderaxespad=0.)

#            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#            plt.savefig(perf_dir+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'.png',bbox_inches='tight')

            plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_'+str(int(base_ps))+'.png',bbox_inches='tight')
            i=i+1
    #########################################################
                g1=np.linspace(0,5,100)
                g=np.power(10,g1)
                num_tasks=np.ceil(ps/g)
                L=np.ceil(num_tasks/th)
                w_c=L*g
                if th==1:
                    w_c=ps*np.ones_like(g)
                for q in range(np.shape(num_tasks)[0]):
                    if num_tasks[q]%th==1 and ps%g[q]!=0:
        #                    w_c_1=problem_size+(1-th)*(L-1)*grain_size
                        w_c[q]=(L[q]-1)*g[q]+(ps%g[q])
                plt.figure(i)
                plt.axes([0, 0, 3, 2])
                plt.scatter(g,(w_c-ps/th)/(ps/th))
                
                
                Lmax=int(np.max(L))
                gs=[]
                for q in range(2,Lmax+1):                
                    g1=math.ceil(ps/((q)*th))
                    if g1 not in gs:
                        plt.axvline(g1,linestyle='dashed',color='green',alpha=0.4)
                        gs.append(g1)
                    g2=math.ceil(ps/((q-1)*th))                
                    if g2 not in gs:
                        plt.axvline(g2,linestyle='dashed',color='green',alpha=0.4)
                        gs.append(g2)
                plt.xlabel('Grain size')
                plt.ylabel('Imbalance ratio')
                plt.xscale('log')
                plt.savefig(perf_dir+'/imbalance_ratio_'+str(int(ps))+'_'+str(int(th))+'.png',bbox_inches='tight')

                
                
                
                plt.figure(i)
                plt.axes([0, 0, 3, 2])
                plt.scatter(new_array[:,5],(new_array[:,-2]-ps/th)/(ps/th),label='imbalance ratio')
                plt.scatter(new_array[:,5],(ps)/new_labels,label='speed-up')
    
                
                
                plt.xlabel('Grain size')
#                plt.ylabel('Imbalance ratio')
                plt.xscale('log')
                plt.legend(bbox_to_anchor=(0.01, 0.95), loc=2, borderaxespad=0.)

                plt.savefig(perf_dir+'/imbalance_ratio_speedup_'+str(int(ps))+'_'+str(int(th))+'.png',bbox_inches='tight')
    ####################################################################            
                if np.shape(new_array[new_array[:,3]>0])[0]>30:
                    plt.figure(i)
    #                plt.axes([0, 0, 1.5, 1])
    
                    z_5=my_func(new_array,*popt_5)
                    test_errors[node][base_ps][ps][th]=100*np.mean(np.abs(z_5-new_labels)/new_labels)
                    r2_errors[node][base_ps][ps][th]=r2_score(new_labels,z_5)
            th=8
            lb=0.5
            ls=0.1
            for ps in base_pss:#[ps for ps in problem_sizes]:
                plt.figure(i)
                print(base_ps, ps)
                array_ps=train_set[train_set[:,0]==ps]
                labels_ps=train_labels[train_set[:,0]==ps]
                
                a_s=np.argsort(array_ps[:,5])
                for ir in range(np.shape(array_ps)[1]):
                    array_ps[:,ir]=array_ps[a_s,ir]
                labels_ps=labels_ps[a_s] 
              
                new_array=array_ps[array_ps[:,2]==th]
                new_labels=labels_ps[array_ps[:,2]==th]
                plt.axes([0, 0, 2, 2])

#                opt=np.logical_and(new_array[:,5]>100, new_array[:,5]<1e6)
#                plt.scatter(new_array[:,5][opt],new_labels[opt],marker='.',label='true')
#                plt.scatter(new_array[:,5],new_labels,marker='.',label=str(int(th))+' threads')
                plt.scatter(new_array[:,5],new_labels,label='true')
                z_5=my_func(new_array,*popt_5)

                plt.scatter(new_array[:,5],z_5,label='prediction')

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
#                plt.axvline(g3,color='green',alpha=1,label='$\lambda_b=0.8$') 
#                plt.axvline(g5,color='green',alpha=0.6,label='$\lambda_b=0.4$')  
#                plt.axvline(g1,color='green',alpha=0.3,label='$\lambda_b=0.1$')  
#
#
#                plt.axvline(g2,color='red',alpha=0.3,label='$\lambda_s=0.1$')    
#                plt.axvline(g6,color='red',alpha=0.6,label='$\lambda_s=0.4$')    
#                plt.axvline(g4,color='red',alpha=1,label='$\lambda_s=0.8$')    
                plt.axvspan(g1,g2,color='green',alpha=0.5)
#                plt.axvspan(g3,g2,color='green',alpha=0.4)
#                plt.axvspan(g2,g4,color='green',alpha=0.4)

#                plt.fill_between(new_array[:,5],where=np.logical_and(new_array[:,5]<=g2,new_array[:,5]>=g1),facecolor='green',alpha=.5)
                plt.xlabel('Grain size')
                plt.ylabel('Execution Time($\mu{sec}$)')
                plt.xscale('log')
#                    print(lb,ls,g1,g2)
#                plt.title('problem size:'+str(int(ps))+'  '+str(int(th))+' threads')
#                plt.axvline(ps/(th),color='gray',linestyle='dotted')  
                plt.legend(bbox_to_anchor=(0.08, 0.98), loc=2, borderaxespad=0.)

#                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#                    plt.savefig(perf_dir+'nows_new_rostam/'+str(int(ps))+'_'+str(int(th))+'_1_all.png',bbox_inches='tight')
#                plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_'+str(int(base_ps))+'.png',bbox_inches='tight')
#                    plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/'+node+'_'+str(int(ps))+'.png',bbox_inches='tight')

                i=i+1                    
#                plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_range_10_50_80.png',bbox_inches='tight')

                plt.savefig(perf_dir+'/fitted/'+str(int(base_ps))+'/ranges/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_range_'+str(int(100*lb))+'_'+str(int(100*ls))+'.png',bbox_inches='tight')

for node in nodes:
    for base_ps in test_errors[node].keys():
#        fig, ax = plt.subplots()
#        plt.axes([0, 0, 3, 1])

        fig=plt.figure(i)
        ax = fig.add_subplot(111)
        width=0.25
        rects1 = ax.bar(threads[node],[test_errors[node][base_ps][base_ps][i] for i in threads[node]], width,label='Base problem size='+str(int(base_ps)))
        #rects2 = ax.bar(np.array([0,1,2])+width,[param_errors[i][1] for i in range(3)], width, color='seagreen',label='test')
        plt.xlabel('Number of cores')
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
        rects1 = ax.bar(threads[node]-.25+width*k, [r2_errors[node][base_ps][base_ps][i] for i in threads[node]], width,label='ps='+str(int(base_ps)))
        k=k+1
        ax.set_xlabel('Number of cores')
        plt.ylabel('R2 score(%)')
        plt.xticks(threads[node])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig=plt.figure(i)
    plt.savefig(perf_dir+'/fitted/'+node+'_r2_error_compared_self.png',bbox_inches='tight')
    i=i+1    
    
for node in nodes:
    k=0
    for base_ps in test_errors[node].keys():
        fig=plt.figure(i)

        ax = fig.add_subplot(111)
        width=0.15
        rects1 = ax.bar(threads[node]-.25+width*k, [test_errors[node][base_ps][base_ps][i] for i in threads[node]], width,label='ps='+str(int(base_ps)))
        k=k+1
        ax.set_xlabel('Number of cores')
        plt.ylabel('Relative Error(%)')
        plt.xticks(threads[node])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig=plt.figure(i)
    plt.savefig(perf_dir+'/fitted/'+node+'_relative_error_compared_self.png',bbox_inches='tight')
    i=i+1        
    

p_th={}

for node in nodes:
    p_th[node]={}
    for base_ps in test_errors[node].keys():
        p_th[node][base_ps]={}        
        for th in threads[node]:
            if node=='tameshk':
                p_th[node][base_ps][th]=np.mean([r2_errors[node][base_ps][ps][th] for ps in r2_errors[node][base_ps].keys() if th in r2_errors[node][base_ps][ps].keys() and ps!=base_ps and ps!=1e8])
            else:
                p_th[node][base_ps][th]=np.mean([r2_errors[node][base_ps][ps][th] for ps in r2_errors[node][base_ps].keys() if th in r2_errors[node][base_ps][ps].keys() and ps!=base_ps])

    k=0
    width = 0.15   
    fig=plt.figure(i)
    ax = fig.add_subplot(111)
    for base_ps in test_errors[node].keys():    
        rects1 = ax.bar(threads[node]-.25+width*k, [p_th[node][base_ps][th] for th in threads[node]], width,label='ps='+str(int(base_ps)))
        k=k+1
        ax.set_xlabel('Number of cores')
        plt.ylabel('R2 score(%)')
        plt.xticks(threads[node])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1
    plt.savefig(perf_dir+'/fitted/'+node+'_r2_error_compared_rest.png',bbox_inches='tight')
    
    
    

p_th={}

for node in nodes:
    p_th[node]={}
    for base_ps in test_errors[node].keys():
        p_th[node][base_ps]={}        
        for th in threads[node]:
            if node=='tameshk':
                p_th[node][base_ps][th]=np.mean([test_errors[node][base_ps][ps][th] for ps in test_errors[node][base_ps].keys() if th in test_errors[node][base_ps][ps].keys() and ps!=base_ps and ps!=1e8])
            else:
                p_th[node][base_ps][th]=np.mean([test_errors[node][base_ps][ps][th] for ps in test_errors[node][base_ps].keys() if th in test_errors[node][base_ps][ps].keys() and ps!=base_ps])

    k=0
    width = 0.15   
    fig=plt.figure(i)
    ax = fig.add_subplot(111)
    for base_ps in test_errors[node].keys():    
        rects1 = ax.bar(threads[node]-.25+width*k, [p_th[node][base_ps][th] for th in threads[node]], width,label='ps='+str(int(base_ps)))
        k=k+1
        ax.set_xlabel('Number of cores')
        plt.ylabel('Relative Error(%)')
        plt.xticks(threads[node])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    i=i+1
    plt.savefig(perf_dir+'/fitted/'+node+'_relative_error_compared_rest.png',bbox_inches='tight')
        
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


#blazemark

b_filename='/home/shahrzad/repos/Blazemark/data/data_perf_all.csv'
marvin_gdir='/home/shahrzad/repos/Blazemark/data/final/grain_size/marvin/general/'
medusa_gdir='/home/shahrzad/repos/Blazemark/data/final/grain_size/medusa/general/'
popt=gf.find_model_parameters([marvin_gdir,medusa_gdir])

def my_func_usl(ndata,kappa,epsilon): 
    alpha=popt[node][0]
    gamma=popt[node][1]
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
    ts=ps
    return alpha*L+ts*(1+(gamma)*(M-1)+(kappa)*(M)*(M-1))*(w_c)/ps+epsilon*(m)

def my_func_usl(ndata,alpha,gamma): 
    alpha=popt[node][0]
    gamma=popt[node][1]
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
    ts=ps
    return alpha*L+ts*(1+(gamma)*(M-1))*(w_c)/ps

marvin_dir='/home/shahrzad/repos/Blazemark/data/final/blazemark/general/marvin/'
medusa_dir='/home/shahrzad/repos/Blazemark/data/final/blazemark/general/medusa/'
bf.write_to_file([marvin_dir,medusa_dir],b_filename)
titles=['runtime','node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','work_per_core','w1','w2','w3','w4','w5','w6','w7','w8','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','set_associativity','datatype','cost','simd_size','execution_time','num_tasks','mflops','include']

dataframe = pandas.read_csv(b_filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[3:]:
    dataframe[col] = dataframe[col].astype(float)
nodes=dataframe['node'].drop_duplicates().values

runtime='hpx'
benchmark='dmatdmatadd'

benchmark='dmatdmatdmatadd'
node='marvin'  
threads={}
g_params={}
test_errors={}
r2_errors={}
chunk_sizes={}
b='4-256'
popt_all={}
true_values={}
pred_values={}

lb=0.01
ls=.1

n=3
if benchmark=='dmatdmatdmatadd':
    n=4
n1=np.ceil(np.sqrt(20*1024*1024/(8*n)))
n2=np.ceil(np.sqrt(28160*1024/(8*n)))
cache_limit={'marvin':n1, 'medusa':n2}
i=1
for node in nodes:
    popt_all[node]={}
    
    true_values[node]={}
    pred_values[node]={}
    chunk_sizes[node]={}
    g_params[node]={}
    threads[node]={}
    test_errors[node]={}
    r2_errors[node]={}
    included=dataframe['include']==1
    node_selected=dataframe['node']==node
    df_n_selected=dataframe[node_selected & included]
    benchmark_selected=dataframe['benchmark']==benchmark
    rt_selected=dataframe['runtime']==runtime
    num_threads_selected=dataframe['num_threads']<=8
    df_nb_selected=df_n_selected[benchmark_selected & num_threads_selected & rt_selected]         
    block_selected_r=df_nb_selected['block_size_row']==4
    block_selected_c=df_nb_selected['block_size_col']==256
    df_nb_selected=df_nb_selected[ block_selected_r & block_selected_c]
    
    matrix_sizes=df_nb_selected['matrix_size'].drop_duplicates().values
    matrix_sizes.sort()
    thr=df_nb_selected['num_threads'].drop_duplicates().values
    thr.sort()
    threads[node][benchmark]=thr
    
    features=['chunk_size','num_blocks','num_threads','grain_size','block_size_row','block_size_col','work_per_core','num_tasks','execution_time']
#    for th in thr:               
#        test_errors[node][th]=[]
#        true_values[node][th]=[]
#        pred_values[node][th]=[]
    for m in [m for m in matrix_sizes]:# if m<cache_limit[node]]:    
        popt_all[node][m]={}
        chunk_sizes[node][m]={}
        chunk_sizes[node][m][str(lb)+'_'+str(ls)]={}

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
            
        def my_func_u(ndata,kappa): 
            alpha,gamma=popt[node]
            N=ndata[:,2]
            n_t=ndata[:,-1]
            M=np.minimum(n_t,N) 
            L=np.ceil(n_t/(M))
            w_c=ndata[:,-2]
            ps=mflop
            return alpha*L+ts*(1+(gamma)*(M-1)+(kappa)*M*(M-1))*(w_c)/ps#+(1)*(d*ps)*np.exp(-((g-ps/N)/(k))**2)#+(1+(gamma)*(M-1))*(w_c)#+(1)*(1/(np.sqrt(2*np.pi)*(d)))*np.exp(-((g-dN)/(ps/N))**2)
        
        m_selected=df_nb_selected['matrix_size']==m
        features=['chunk_size','num_blocks','num_threads','grain_size','block_size_row','block_size_col','work_per_core','num_tasks','execution_time']
        df_selected=df_nb_selected[m_selected][features]
    
        array_b=df_selected.values
        array_m=array_b[:,0:-1]
        labels_m=array_b[:,-1]
        
        param_bounds=([0],[np.inf])
        popt_c, pcov=curve_fit(my_func_u,array_m,labels_m,method='trf',bounds=param_bounds)
        popt_all[node][m]=popt_c
        
        g_params[node][benchmark]=bf.grain_dict(array_b,1)
    
        for th in thr:

            new_array=array_b[array_b[:,2]==th][:,:-1]
            new_labels=array_b[array_b[:,2]==th][:,-1]
            ts=g_params[node][benchmark][mflop][1][0]
            

            zb=bf.my_model_b(new_array,*popt[node],mflop,ts)
            zu=bf.my_model_u(new_array,*popt[node],*popt_c,m,mflop,ts)
            g1=np.ceil(np.sqrt(popt[node][0]*mflop/(th*lb)))
            g2=np.floor(mflop/(th*(1+np.ceil(1/ls))))
            plt.figure(i)
#            plt.axes([0, 0, 2, 2])
            plt.scatter(new_array[:,3],new_labels,color='blue',label='true')
            plt.scatter(new_array[:,3],zb,color='red',label='prediction')
            plt.scatter(new_array[:,3],zu,color='green',label='full prediction')

            plt.axvline(mflop/th,color='gray',linestyle='dashed')
#            plt.axvspan(g1,g2,color='green',alpha=0.5)
            plt.ylabel('Execution Time($\mu{sec}$)')
            plt.xlabel('Grain size')
            plt.xscale('log')
#            plt.legend(bbox_to_anchor=(0.08, 0.98), loc=2, borderaxespad=0.)

#            plt.savefig(perf_dir+'/blazemark/'+node+'_'+benchmark+'_'+str(int(m))+'_'+str(int(th))+'_range_'+str(int(100*lb))+'_'+str(int(100*ls))+'.png',bbox_inches='tight')

#            plt.savefig(perf_dir+'/blazemark/'+node+'_'+benchmark+'_'+str(int(m))+'_'+str(int(th))+'.png',bbox_inches='tight')
            i=i+1
            
            
            
            
            
            
            
            
            
            
            
            
            [test_errors[node][th].append(k) for k in np.abs(zb-new_labels)/new_labels]
            r2_errors[node][th]=r2_score(new_labels,zb)
            [true_values[node][th].append(k) for k in new_labels]
            [pred_values[node][th].append(k) for k in zb]

    for node in nodes:
        plt.figure(i)
        plt.axes([0, 0, 2, 2])
        error={}
        for th in thr:
            tv=true_values[node][th]
            pv=pred_values[node][th]
            error[th]=100*sum([abs(tv[k]-pv[k])/tv[k] for k in range(len(tv))])/len(tv)

        plt.bar(thr,[error[th] for th in thr])
        plt.xlabel('Number of cores')
        plt.ylabel('Relative Error(%)')
        plt.xticks(thr)      
        plt.savefig(perf_dir+'/blazemark/'+node+'_'+benchmark+'_relative_error_limit.png',bbox_inches='tight')
        i=i+1
        
        
    for node in nodes:
        plt.figure(i)
        plt.axes([0, 0, 2, 2])
        r2={}
        for th in thr:
            tv=true_values[node][th]
            pv=pred_values[node][th]
            r2[th]=r2_score(tv,pv)
        plt.bar(thr,[r2[th] for th in thr])

        plt.xlabel('Number of cores')
        plt.ylabel('R2 score(%)')
        plt.xticks(thr)      
        plt.savefig(perf_dir+'/blazemark/'+node+'_'+benchmark+'_r2_score_all.png',bbox_inches='tight')
        i=i+1
    
           