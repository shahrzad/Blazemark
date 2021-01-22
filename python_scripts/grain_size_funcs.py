#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 23:52:07 2020

@author: shahrzad
"""

import csv
import glob
import numpy as np
import pandas
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 22})
from scipy.optimize import curve_fit

def create_dict(directories,data_filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv', mini=False, maxi=False):
    to_csv=True
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

    for filenames in data_files:
        filename=filenames.replace('marvin_old','c7')
        filename=filename.replace('_all_multiple_tasks','')
        filename=filename.replace('_idle_mask','')
        filename=filename.replace('qs_idle','')
        filename=filename.replace('_idle','')
        filename=filename.replace('_all','')
        
        if 'seq' in filename:
            (node, _,_, _, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
            chunk_size=0
            th=1         
        elif len(filename.split('/')[-1].replace('.dat','').split('-'))==6:
            (node, _,_, th, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
            chunk_size=1
        else:
            (node, _,_, th, chunk_size, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
        if 'spt' in filename:
            chunk_size=-1
            
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
                                                           
    data_files.sort()   
    problem_sizes=[]

    for filenames in data_files:   
        f=open(filenames, 'r')
                 
        result=f.readlines()
        filename=filenames.replace('marvin_old','c7')
        filename=filename.replace('_all_multiple_tasks','')
        filename=filename.replace('_idle_mask','')
        filename=filename.replace('qs_idle','')
        filename=filename.replace('_idle','')
        filename=filename.replace('_all','')
        if len(result)!=0:
            avg=0
            if mini or maxi:
                avg=[]
            if 'seq' in filename:
                (node, _,_, _, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
                chunk_size=0
                th=1   
            elif len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
                (node, _,_, th, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
                chunk_size=1
            else:
                (node, _,_, th, chunk_size, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
            if 'spt' in filename:
                chunk_size=-1
            chunk_size=float(chunk_size)        
            th=float(th)       
            iter_length=float(iter_length)
            num_iteration=float(num_iteration)    
            first=True
            for r in [r for r in result if r!='\n' and not r.startswith('split type:')]:  
                if not first:
                    if mini or maxi:
                        avg.append(float(r.split('in ')[1].split('microseconds')[0].strip()))
                    else:
                        avg+=float(r.split('in ')[1].split('microseconds')[0].strip())
                else:
                    first=False
            if mini:
                avg=min(avg)
            elif maxi:
                avg=max(avg)
            else:
                avg=avg/(len([r for r in result if r!='\n' and not r.startswith('split type:')])-1)
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


def create_spt_dict(spt_filename,iteration_length=1):
    titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
    
    dataframe = pandas.read_csv(spt_filename, header=0,index_col=False,dtype=str,names=titles)
    for col in titles[1:]:
        dataframe[col] = dataframe[col].astype(float)
    
    nodes=dataframe['node'].drop_duplicates().values
    nodes.sort()
    
    node=nodes[0]
    node_selected=dataframe['node']==node
    iter_selected=dataframe['iter_length']==iteration_length
    th_selected=dataframe['num_threads']>=1
    cs_selected=dataframe['chunk_size']==-1
    
    df_n_selected=dataframe[node_selected & cs_selected & iter_selected & th_selected][titles[1:]]
    
    thr=df_n_selected['num_threads'].drop_duplicates().values
    thr.sort()
    
    problem_sizes=df_n_selected['problem_size'].drop_duplicates().values
    problem_sizes.sort()
    
    spt_results={}
    node=nodes[0].replace('-spt','')

    spt_results[node]={}
    
    array=df_n_selected.values
    for ps in problem_sizes:
        array_ps=array[array[:,0]==ps]
        spt_results[ps]={}
        for th in thr:
            array_t=array_ps[array_ps[:,2]==th]
            spt_results[ps][th]=array_t[:,-1]
    return spt_results

#def get_split_info_idle(directories):
#    thr=[]
#    data_files=[]
#    
#    for directory in directories:
#        [data_files.append(i) for i in glob.glob(directory+'/*.dat')]
#    
#    chunk_sizes=[]
#    num_iterations=[]
#    iter_lengths=[]
#    nodes=[]
#    split_info={}
#    for filenames in data_files:
#        filename=filenames.replace('marvin_old','c7')
#        filename=filename.replace('_idle','')
#        filename=filename.replace('_all','')
#        
#        if 'seq' in filename:
#            (node, _,_, _, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
#            chunk_size=0
#            th=1         
#        elif len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
#            (node, _,_, th, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
#            chunk_size=1
#        else:
#            (node, _,_, th, chunk_size, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
#        if 'spt' in filename:
#            chunk_size=-1
#            
#        chunk_size=int(chunk_size)
#        th=int(th)
#        iter_length=int(iter_length)
#        num_iteration=int(num_iteration)
#        if node not in nodes:
#            nodes.append(node)
#        if iter_length not in iter_lengths:
#            iter_lengths.append(iter_length)
#        if num_iteration not in num_iterations:
#            num_iterations.append(num_iteration)  
#            split_info[num_iteration]={}
#        if th not in thr:
#            thr.append(th)
#        if chunk_size not in chunk_sizes:
#            chunk_sizes.append(chunk_size)
#
#    nodes.sort()
#    num_iterations.sort()
#    iter_lengths.sort()
#    thr.sort()                  
#                                                           
#    data_files.sort()   
#
#    for filenames in data_files:   
#        f=open(filenames, 'r')
#                 
#        result=f.read()
#
#        filename=filenames.replace('marvin_old','c7')
#        filename=filename.replace('_idle','')
#        filename=filename.replace('_all','')
#
#        if len(result)!=0:
#            if 'seq' in filename:
#                (node, _,_, _, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
#                chunk_size=0
#                th=1   
#            elif len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
#                (node, _,_, th, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
#                chunk_size=1
#            else:
#                (node, _,_, th, chunk_size, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
#            if 'spt' in filename:
#                chunk_size=-1
#            chunk_size=float(chunk_size)        
#            th=float(th)       
#            iter_length=float(iter_length)
#            num_iteration=float(num_iteration)    
#           
#            split_info[num_iteration][th]=[]
#            for r in result.split('\n\n')[:-1]:
#                rps=r.split('num_iterations')[0].split(' task index:')[1:]
#                num_tasks=len(rps)
#                split_info[num_iteration][th].append(num_tasks)
#                
#                
##                for rp in rps:
##                    task_id=int(rp.split(' from: ')[0].strip())
##                    start_id=int(rp.split(' from: ')[1].split(' to:')[0].strip())
##                    if task_id==0:
##                        index=index+1
##                        split_info[num_iteration][th][index]={}
##                    split_info[num_iteration][th][index][task_id]=[start_id]
##                   
#    return split_info

def get_task_info(directories, save_dir_name, plot=False, save=False, value_sorted=False, plot_reps=False, iteration_length=1):
    perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'

    thr=[]
    data_files=[]
    counters=['/time/average', '/count/cumulative', '/time/cumulative']
    for directory in directories:
        [data_files.append(i) for i in glob.glob(directory+'/*.dat')]
    
    chunk_sizes=[]
    num_iterations=[]
    iter_lengths=[]
    nodes=[]    
    iter_files=[]
    
    for filenames in data_files:
        filename=filenames.replace('marvin_old','c7')
        filename=filename.replace('ctr_all_multiple_tasks','')
        filename=filename.replace('ctr_idle_mask','')
        filename=filename.replace('ctr_idle','')
        filename=filename.replace('ctr_all','')
        
        if 'seq' in filename:
            (node, _,_, _, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
            chunk_size=0
            th=1         
        elif len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
            (node, _,_, th, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
            chunk_size=1
        else:
            (node, _,_, th, chunk_size, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
#        if 'spt' in filename:
#            chunk_size=-1

        iter_length=int(iter_length)
        if iter_length==iteration_length: 
            iter_files.append(filenames)                       
            chunk_size=int(chunk_size)
            th=int(th)
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
    th_data={}
              
    for num_iteration in num_iterations:
        th_data[num_iteration]={}
        for task_thresh in chunk_sizes:
            th_data[num_iteration][task_thresh]={}
            for th in thr:
                th_data[num_iteration][task_thresh][th]={}
                                             
    data_files.sort()   

    for filenames in iter_files:   
        f=open(filenames, 'r')
                 
        result=f.read()

        filename=filenames.replace('marvin_old','c7')
        filename=filename.replace('ctr_all_multiple_tasks','')
        filename=filename.replace('ctr_idle_mask','')
        filename=filename.replace('ctr_idle','')
        filename=filename.replace('ctr_all','')
        
        if len(result)!=0:
            if 'seq' in filename:
                (node, _,_, _, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
                chunk_size=0
                th=1   
            elif len(filename.split('/')[-1].replace('.dat','').split('_'))==6:
                (node, _,_, th, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
                chunk_size=1
            else:
                (node, _,_, th, chunk_size, iter_length, num_iteration) = filename.split('/')[-1].replace('.dat','').split('_')         
#            if 'spt' in filename:
#                chunk_size=-1
            task_thresh=int(chunk_size)        
            th=int(th)       
            iter_length=int(iter_length)
            num_iteration=int(num_iteration)    
            th_data[num_iteration][task_thresh][th]={}
            perf_data=[rs.split('Initialization')[0] for rs in result.split('Done')[1:]]
            th_data[num_iteration][task_thresh][th]['exec_time']={}
            for counter in counters:
                th_data[num_iteration][task_thresh][th][counter]={}
            for ind,pd in enumerate(perf_data):
                th_data[num_iteration][task_thresh][th]['exec_time'][ind]=float(pd.split('splittable_executor in ')[1].split(' microseconds')[0])*1000
                for counter in counters:
                    th_data[num_iteration][task_thresh][th][counter][ind]=[]
                    for t in np.arange(th):
                        if counter=='/count/cumulative':
                            count=int(pd.split('/threads{locality#0/pool#default/worker-thread#'+str(t)+'}'+counter+',')[1].split('\n')[0].split(',')[-1])
                            th_data[num_iteration][task_thresh][th][counter][ind].append(count)
                        elif counter=='/time/average':
                            time=float(pd.split('/threads{locality#0/pool#default/worker-thread#'+str(t)+'}'+counter+',')[1].split('\n')[0].split(',')[-2])
                            th_data[num_iteration][task_thresh][th][counter][ind].append(time)
                        elif counter=='/time/cumulative':
                            time=float(pd.split('/threads{locality#0/pool#default/worker-thread#'+str(t)+'}'+counter+',')[1].split('\n')[0].split(',')[-2])
                            th_data[num_iteration][task_thresh][th][counter][ind].append(time)
                                   
    if plot:           
        k=1
        i=1         
        for ps in th_data.keys():
            for counter in [c for c in counters if c!='/time/average']:
                for th in thr:
                    cores = np.arange(th+1)
                    for rep in range(len(perf_data)): 
                        width = 0.3     
                        if plot_reps:
                            if len(chunk_sizes)>1:
                                print('error')
                                break
                            
                            labels=[str(t) for t in np.arange(th)]
                            task_thresh=chunk_sizes[0]
                            plt.figure(k)
                            y=[th_data[ps][task_thresh][th][counter][rep][t] for t in np.arange(th)]
                            if value_sorted==True:
                                y.sort()
                            if 'count' in counter:
                                y.append((ps*1e3)/th_data[ps][task_thresh][th]['exec_time'][rep])
                                labels.append('speed up')
                                plt.ylabel(counter)
                            elif 'time' in counter:
                                y.append(th_data[ps][task_thresh][th]['exec_time'][rep])
                                labels.append('exec time')
                                plt.ylabel(counter+'[ns]')
                            plt.bar(cores+rep*width, y, width, label='min_task_size:'+str(task_thresh)+' run '+str(rep+1))
                            plt.title('problem size: '+str(int(ps))+' '+str(int(th))+' threads  iteration length:'+str(int(iteration_length)))
                            plt.xlabel('Core #')                                   
                            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                            plt.xticks(cores+rep*width / 2, labels)
                        else:
                            for j,task_thresh in enumerate(chunk_sizes):
                                labels=[str(t) for t in np.arange(th)]
    
                                plt.figure(i)
                                y=[th_data[ps][task_thresh][th][counter][rep][t] for t in np.arange(th)]
                                if value_sorted==True:
                                    y.sort()
                                if 'count' in counter:
                                    y.append((ps*1e3)/th_data[ps][task_thresh][th]['exec_time'][rep])
                                    labels.append('speed up')
                                    plt.ylabel(counter)
                                elif 'time' in counter:
                                    y.append(th_data[ps][task_thresh][th]['exec_time'][rep])
                                    labels.append('exec time')
                                    plt.ylabel(counter+'[ns]')
                                plt.bar(cores+j*width, y, width, label='min_task_size:'+str(task_thresh))
                                plt.title('problem size: '+str(int(ps))+' '+str(int(th))+' threads'+'  run '+str(int(rep)+1)+'  iteration length:'+str(int(iteration_length)))
                                plt.xlabel('Core #')                                   
                                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                                plt.xticks(cores+j*width / 2, labels)

#                            plt.legend(loc='best')

                        if save and not plot_reps:
                            plt.figure(i)
                            counter_type=counter[1:].replace('/','-')
                            plt.savefig(perf_dir+'/splittable/'+save_dir_name+'/'+counter_type+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_'+str(int(rep))+'.png',bbox_inches='tight')
                            plt.close()
                        i=i+1
                    if save and plot_reps:
                        plt.figure(k)
                        counter_type=counter[1:].replace('/','-')
                        plt.savefig(perf_dir+'/splittable/'+save_dir_name+'/'+counter_type+'/'+node+'_'+str(int(ps))+'_'+str(int(th))+'_'+str(int(rep))+'.png',bbox_inches='tight')
                        plt.close()
                        k=k+1
    return th_data
#            
#            for i in range(len(result)):
#                rn=result[i]
#                if rn.startswith('number of idle cores: '):
#                    num_idle=int(rn.split('cores: ')[1].split('remaining')[0])
#                    r=result[i+1]
#                    r=r.replace('number',' ')
#                    [task_id,start_id,end_id]=[int(ir) for ir in r.split() if ir.isdigit()] 
#                    if task_id==0:
#                        index=index+1
#                        split_info[num_iteration][th][index]={}
#                    split_info[num_iteration][th][index][task_id]=[num_idle,start_id,end_id]
#                   
                
           
def compare_results(dirs, save_dir_name, alias=None, save=True, mode='ps-th', iteration_lengths=None):
#    perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'

    spt_results={}
    descs=[]
    for i in range(len(dirs)):
        directory=dirs[i]
        spt_filename='/home/shahrzad/repos/Blazemark/data/data_grain_spt_tmp.csv'
        create_dict([directory], data_filename=spt_filename)
        if alias is not None:
            desc=alias[i]
        else:
            desc=directory.split('/')[-3].split('_')[0]+'_'+directory.split('/')[-2]
        print(desc)
        il=1
        if iteration_lengths is not None:
            il=iteration_lengths[i]
        spt_results[desc]=create_spt_dict(spt_filename,il)
        descs.append(desc)

#    problem_sizes=[ps for ps in spt_results[descs[0]].keys()]
    
    node=[n for n in spt_results[descs[0]].keys()][0]
    problem_sizes=[ps for ps in spt_results[descs[0]][node].keys()]
    threads={}
    
    titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
    filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'
    
    dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
    for col in titles[1:]:
        dataframe[col] = dataframe[col].astype(float)
        
    node_selected=dataframe['node']=='marvin'
    nt_selected=dataframe['num_tasks']>=1
    iter_selected=dataframe['iter_length']==1
    th_selected=dataframe['num_threads']>=1
    df_n_selected=dataframe[node_selected & nt_selected & iter_selected & th_selected][titles[1:]]
#    
#    problem_sizes=df_n_selected['problem_size'].drop_duplicates().values
#    problem_sizes.sort()
    thr=df_n_selected['num_threads'].drop_duplicates().values
    thr.sort()
    
    array=df_n_selected.values
    if mode=='th':
        plot_th(array, spt_results, thr, save_dir_name, save)
    if mode=='mode':
        plot_mode(spt_results, thr, save_dir_name, save)
    elif mode=='ps-th':
        plot_ps_th(array, problem_sizes, spt_results, thr, save_dir_name, save)
    elif mode=='ps':
        plot_ps(array, spt_results, thr, save_dir_name, save)


def plot_ps_th(array, problem_sizes, spt_results, thr, save_dir_name, save):
    colors=['red', 'green', 'purple', 'pink', 'cyan', 'lawngreen', 'yellow']
    perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'
    i=1
    descs=[desc for desc in spt_results.keys()]
    for ps in [100000, 100000000]:#problem_sizes:
        array_ps=array[array[:,0]==ps]
        if np.shape(array_ps)[0]!=0:
            for th in thr:
                plt.figure(i)
                plt.axes([0, 0, 1.5, 1.5])
                array_t=array_ps[array_ps[:,2]==th]
                plt.scatter(array_t[:,5],array_t[:,-1])
                for c,desc in zip(colors,descs):
#                    plt.figure(i)                
#                    plt.axhline(spt_results[desc][ps][th], color=c, label=desc)
                    plt.figure(i)  
                    node=[n for n in spt_results[desc].keys()][0] 
                    plt.axhline(spt_results[desc][node][ps][th], color=c, label=desc)
           
                plt.axvline(ps/th,color='gray',linestyle='dashed')
                plt.xlabel('Grain size')
                plt.ylabel('Execution Time($\mu{sec}$)')
                plt.xscale('log')
#                plt.title('ps='+str(int(ps))+' '+str(int(th))+' threads')
                plt.legend(bbox_to_anchor=(0.08, 0.98), loc=2, borderaxespad=0.)
                i=i+1
                if save:
#                    plt.savefig(perf_dir+'/splittable/'+save_dir_name+'/'+str(int(ps))+'_'+str(int(th))+'.png',bbox_inches='tight')
                    plt.savefig(perf_dir+'/'+save_dir_name+'/splittable/'+node+'_'+str(int(ps))+'_'+str(int(th))+'.png',bbox_inches='tight')
    

def plot_mode(spt_results, thr, save_dir_name, save):
    perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'
    i=1
    for desc in spt_results.keys():
        for th in thr:
#            plt.figure(i)                
#            plt.scatter([ps for ps in spt_results[desc].keys()],[ps/spt_results[desc]['c7-spt'][ps][th] for ps in spt_results[desc]['c7-spt'].keys()], label=str(int(th))+' threads', marker='.')
            plt.figure(i)  
            node=[n for n in spt_results[desc].keys()][0] 
              
            plt.scatter([ps for ps in spt_results[desc][node].keys()],[ps/spt_results[desc][node][ps][th] for ps in spt_results[desc][node].keys()], label=str(int(th))+' threads', marker='.')
   
        plt.xlabel('Problem size')
        plt.ylabel('Execution time/ps')
#        plt.xscale('log')
        plt.title('mode: '+desc)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        i=i+1
        if save:
            plt.savefig(perf_dir+'/splittable/'+save_dir_name+'/'+node+'_'+str(int(th))+'.png',bbox_inches='tight')


def plot_th(array, spt_results, thr, save_dir_name, save):
    perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'
    node='marvin_old'
    colors=['red', 'cyan', 'lawngreen', 'green', 'purple', 'pink', 'yellow']
    i=1
    descs=[desc for desc in spt_results.keys()]
    problem_sizes=[ps for ps in spt_results[descs[0]]['c7-spt'].keys()]
    for th in thr:
        array_t=array[array[:,2]==th]
        t_min_execs=[]
        t_equals=[]
        for ps in problem_sizes:
            array_ps=array_t[array_t[:,0]==ps]
            if np.shape(array_ps)[0]!=0:
                l=np.argmin(abs(array_ps[:,5]-ps/th))
                t_equals.append(ps/array_ps[l,-1])
                t_min_execs.append(ps/np.min(array_ps[:,-1]))
            
        for c,desc in zip(colors,spt_results.keys()):
#            plt.figure(i)                            
#            plt.scatter(problem_sizes,[ps/spt_results[desc][ps][th] for ps in spt_results[desc]['c7-spt'].keys()], color=c,label=desc, marker='.')
            plt.figure(i)                        
            node=[n for n in spt_results[desc].keys()][0] 

            plt.scatter(problem_sizes,[ps/spt_results[desc][node][ps][th] for ps in spt_results[desc][node].keys()], color=c,label=desc, marker='.')
        plt.scatter(problem_sizes,t_min_execs, label='best', marker='*')
        plt.scatter(problem_sizes,t_equals, label='equal', marker='+')

        plt.xlabel('Problem size')
        plt.ylabel('Speed-up')
        plt.xscale('log')
        plt.title(str(int(th))+' threads')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        i=i+1
        if save:
            plt.savefig(perf_dir+'/splittable/'+save_dir_name+'/'+node+'_'+str(int(th))+'.png',bbox_inches='tight')

def plot_ps(array, spt_results, thr, save_dir_name, save):
    perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'
    node='marvin_old'
    colors=['red', 'blue', 'cyan', 'lawngreen', 'purple', 'green', 'yellow']
    i=1
    descs=[desc for desc in spt_results.keys()]
    problem_sizes=[ps for ps in spt_results[descs[0]].keys()]
    for ps in problem_sizes:
        array_ps=array[array[:,0]==ps]
        t_min_execs=[]
        t_equals=[]
        for th in thr:
            array_t=array_ps[array_ps[:,2]==th]
            if np.shape(array_t)[0]!=0:
                l=np.argmin(abs(array_t[:,5]-ps/th))
                t_equals.append(ps/array_t[l,-1])
                t_min_execs.append(ps/np.min(array_t[:,-1]))
            
        for c,desc in zip(colors,spt_results.keys()):
            plt.figure(i)                            
            plt.plot(thr,[ps/spt_results[desc][ps][th] for th in thr], color=c,label=desc, marker='.')
        plt.plot(thr,t_min_execs, label='best', marker='*',color=colors[len(descs)],linestyle='dashed')
        plt.plot(thr,t_equals, label='equal', marker='+',color=colors[len(descs)+1],linestyle='dashed')

        plt.xlabel('# cores')
        plt.ylabel('Speed-up')
#        plt.xscale('log')
        plt.title('problem size:'+str(int(ps)))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        i=i+1
        if save:
            plt.savefig(perf_dir+'/splittable/'+save_dir_name+'/'+node+'_'+str(int(ps))+'.png',bbox_inches='tight')


def my_model(ndata,alpha,gamma): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=ndata[:,0]
    ts=ps
    return alpha*L+ts*(1+(gamma)*(M-1))*(w_c)/ps#+(1)*(d*ps)*np.exp(-((g-ps/N)/(k))**2)#+(1+(gamma)*(M-1))*(w_c)#+(1)*(1/(np.sqrt(2*np.pi)*(d)))*np.exp(-((g-dN)/(ps/N))**2)

def find_model_parameters(directories,data_filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'):
    create_dict(directories)
    titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']

    perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general/thesis'
    
    dataframe = pandas.read_csv(data_filename, header=0,index_col=False,dtype=str,names=titles)
    for col in titles[1:]:
        dataframe[col] = dataframe[col].astype(float)
    
    nodes=dataframe['node'].drop_duplicates().values
    nodes.sort()
    popt={}               
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
        popt[node]={}
        base_ps=1e5
        array_selected_ps=array_all[array_all[:,0]==base_ps]
               
        array_ps=array_selected_ps[:,:-1]
        labels_ps=array_selected_ps[:,-1]
            
        a_s=np.argsort(array_ps[:,5])
        for ir in range(np.shape(array_ps)[1]):
            array_ps[:,ir]=array_ps[a_s,ir]
        labels_ps=labels_ps[a_s]     
        
        param_bounds=([0,0],[np.inf,np.inf])
    
        param, pcov=curve_fit(my_model,array_ps,labels_ps,method='trf',bounds=param_bounds)
        popt[node]=param
    return popt
