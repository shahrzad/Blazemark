#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:15:34 2020

@author: shahrzad
"""
import math
import csv
import glob
import numpy as np
import pandas
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

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

def create_dict_reference(directory):
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


def create_spt_dict(filename,benchmark):
    titles=['runtime','node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','work_per_core','w1','w2','w3','w4','w5','w6','w7','w8','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','set_associativity','datatype','cost','simd_size','execution_time','num_tasks','mflops','include']

    dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
    for col in titles[3:]:
        dataframe[col] = dataframe[col].astype(float) 
        
    nodes=dataframe['node'].drop_duplicates().values
    nodes.sort()
    runtime='hpx'
    threads={}
    spt_results={}
    spt_node=nodes[0]
    b='4-256'
    
    threads[spt_node]={}
    included=dataframe['include']==1
    node_selected=dataframe['node']==spt_node
    df_n_selected=dataframe[node_selected & included]
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
    threads[spt_node][benchmark]=thr    
    features=['chunk_size','num_blocks','num_threads','grain_size','block_size_row','block_size_col','work_per_core','num_tasks','execution_time']    
    spt_node=spt_node.replace('_spt','')
    spt_results[spt_node]={}
    spt_results[spt_node][b]={}
    spt_results[spt_node][b][benchmark]={}
    
    for m in matrix_sizes:
        m_selected=df_nb_selected['matrix_size']==m
        spt_results[spt_node][b][benchmark][m]={}
        df_selected=df_nb_selected[m_selected][features]
        array_b=df_selected.values
    
        for th in thr:
            spt_results[spt_node][b][benchmark][m][th]=array_b[array_b[:,2]==th][:,-1]
    
    return spt_results


def create_dict_relative_norepeat(directories):
    thr={}
    data_files=[]
    
    for directory in directories:
        [data_files.append(i) for i in glob.glob(directory+'/*.dat')]
    benchmark=''
    benchmarks=[]
    chunk_sizes=[]
    block_sizes={}
    mat_sizes={}
    nodes=[]
    
    for filenames in data_files:
        filename=filenames.replace('marvin_old','c7')
        filename=filename.replace('qs_idle','')
        filename=filename.replace('_idle','')
        filename=filename.replace('_all','')
        filename=filename.replace('_adaptive','')
        try:
            (node, benchmark, th, runtime, chunk_size, block_size_row, block_size_col, mat_size) = filename.split('/')[-1].replace('.dat','').split('-')         
        except:
            (node, benchmark, th, runtime, block_size_row, block_size_col, mat_size) = filename.split('/')[-1].replace('.dat','').split('-')         
            chunk_size=-1

        mat_size=mat_size.split(',')[0]
        if benchmark not in benchmarks:
                benchmarks.append(benchmark)   
                mat_sizes[benchmark]=[]
                block_sizes[benchmark]=[]
        if node not in nodes:
            thr[node]=[]
        if int(mat_size) not in mat_sizes[benchmark]:
            mat_sizes[benchmark].append(int(mat_size))
        if int(th) not in thr[node]:
            thr[node].append(int(th))        
        if block_size_row+'-'+block_size_col not in block_sizes[benchmark]:
            block_sizes[benchmark].append(block_size_row+'-'+block_size_col)
        if chunk_size=='':
            chunk_size=-1
        if int(chunk_size) not in chunk_sizes:
            chunk_sizes.append(int(chunk_size))
        if node not in nodes:
            nodes.append(node)
                  
    [thr[node].sort() for node in thr.keys()]
    nodes.sort()      
    chunk_sizes.sort()
    benchmarks.sort()   
    
    d={}
    for node in nodes:
        d[node]={}
        for benchmark in benchmarks:  
            mat_sizes[benchmark].sort()
            block_sizes[benchmark].sort()
            d[node][benchmark]={}
           
            for th in thr[node]:
                d[node][benchmark][th]={}
                for bs in block_sizes[benchmark]:
                    d[node][benchmark][th][bs]={}
                    for cs in chunk_sizes:
                        d[node][benchmark][th][bs][cs]={}
                        d[node][benchmark][th][bs][cs]['size']=mat_sizes[benchmark]
                        d[node][benchmark][th][bs][cs]['mflops']=[0]*len(mat_sizes[benchmark])                                                            
    data_files.sort()        
    for filenames in data_files:  
            
        stop=False
        f=open(filenames, 'r')                 
        result=f.readlines()[3:]
        filename=filenames.replace('marvin_old','c7')
        filename=filename.replace('qs_idle','')
        filename=filename.replace('_idle','')
        filename=filename.replace('_all','')  
        filename=filename.replace('_adaptive','')

        try:
            (node, benchmark, th, runtime, chunk_size, block_size_row, block_size_col, mat_size) = filename.split('/')[-1].replace('.dat','').split('-')         
        except:
            (node, benchmark, th, runtime, block_size_row, block_size_col, mat_size) = filename.split('/')[-1].replace('.dat','').split('-')         
            chunk_size=-1
            
        th=int(th)       
        if chunk_size=='':
            chunk_size=-1
        chunk_size=int(chunk_size)         
        for r in result:        
            if "N=" in r:
                stop=True
            if not stop:
                s=mat_sizes[benchmark].index(int(r.strip().split(' ')[0]))
                d[node][benchmark][th][block_size_row+'-'+block_size_col][chunk_size]['mflops'][s]=float(r.strip().split(' ')[-1])

    return (d, chunk_sizes, block_sizes, thr, benchmarks, mat_sizes)  


def write_to_file(directories, filename):
    (d_hpx,  chunk_sizes, block_sizes, thr, benchmarks, mat_sizes)=create_dict_relative_norepeat(directories)
    L1cache='32768'
    L2cache='262144'
    L3cache='20971520'
    cache_line='8'
    set_associativity='512'
    simdsize=4
    f=open(filename,'w')
    f_writer=csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    f_writer.writerow(['runtime','node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','work_per_core','w1','w2','w3','w4','w5','w6','w7','w8','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','set_associativity','datatype','cost','simd_size','execution_time','num_tasks','mflops','include'])
    node_type=0
    for node in d_hpx.keys():
        if node=='marvin' or node=='marvin_old' or node=='c7':
            L1cache='32768'
            L2cache='262144'
            L3cache='20971520'
            cache_line='8'
            set_associativity='512'
            simdsize=4
        elif node=='trillian':
            L1cache='65536'
            L2cache='2097152'
            L3cache='6291456'
            cache_line='16'
            simdsize=4
            set_associativity='131072'
        elif node=='medusa':
            L1cache='32768'
            L2cache='1048576'
            L3cache='28835840'
            cache_line='64'
            set_associativity='16' 
            simdsize=8
        benchmark_type=0
        for benchmark in d_hpx[node].keys():
            for th in [th for th in d_hpx[node][benchmark].keys() if th<=8]:
                for b in d_hpx[node][benchmark][th].keys():
                    for c in d_hpx[node][benchmark][th][b].keys():  
                        for m in mat_sizes[benchmark]:
                            k=d_hpx[node][benchmark][th][b][c]['size'].index(m)
                            if 'mflops' in d_hpx[node][benchmark][th][b][c].keys() and d_hpx[node][benchmark][th][b][c]['mflops'][k]:
                                if not(benchmark=='dmatdmatadd' and (b=='64-64' or b=='8-1024' or b=='4-64')):  
                                    r=d_hpx[node][benchmark][th][b][c]['mflops'][k]
    
                                   
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
                                        
                                    if th==1:
                                        ratio=0
                                    else:
                                        ratio=str(num_blocks/(c*(th-1)))
                                    mflop=0
                                    if benchmark=='dmatdmatadd':                            
                                        mflop=b_r*b_c                            
                                    elif benchmark=='dmatdmatdmatadd':
                                        mflop=b_r*b_c*2
                                    else:
                                        mflop=b_r*b_c*(2*m)
                                        
                                    num_elements=[mflop]*num_blocks
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
                                                                                    
                                    data_type=8
                                    grain_size=sum(num_elements[0:c])
                                    num_mat=3
                                    if benchmark=='dmatdmatdmatadd':
                                        num_mat=4
                                    cost=c*mflop*num_mat/data_type
                                    aligned_m=m
                                    if m%simdsize!=0:
                                        aligned_m=m+simdsize-m%simdsize
                                    if benchmark=='dmatdmatadd':                            
                                        mflop=(aligned_m)*m                          
                                    elif benchmark=='dmatdmatdmatadd':
                                        mflop=2*(aligned_m)*m
                                    else:
                                        mflop=2*(aligned_m)**3        
                                    exec_time=mflop/r
                                    num_tasks=np.ceil(num_blocks/c)
                                    task_sizes=[0.]*int(num_tasks)
                                    wc=[0.]*8
                                    for i in range(int(num_tasks)):
                                        task_sizes[i]=sum(num_elements[i*c:(i+1)*c])
                                    for i in range(th):
                                        wc[i]=sum([task_sizes[j] for j in range(len(task_sizes)) if j%th==i])
                                    work_per_core=max(wc)
                                    
                                    include=1
                                    if num_mat*b_r*b_c*8>int(L2cache):
                                        include=0
                                    f_writer.writerow(['hpx',node,benchmark,str(m),str(th),b.split('-')[0], 
                                                       b.split('-')[1], str(b_r * b_c), str(work_per_core),
                                                       str(wc[0]),str(wc[1]),str(wc[2]),str(wc[3]),
                                                       str(wc[4]),str(wc[5]),str(wc[6]),str(wc[7]),
                                                       str(c),str(grain_size),str(num_blocks), str(num_blocks/c),
                                                       str(b_r * b_c*c),str(num_blocks/th),ratio,L1cache,L2cache,L3cache,cache_line,set_associativity,str(data_type),str(cost),str(simdsize),str(exec_time),str(num_tasks),r,str(include)])
            benchmark_type+=1
        node_type+=1        
    f.close()                    

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx] 

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
def compare_results(dirs, save_dir_name, benchmark, alias=None, save=True, ref=False, plot=True, plot_bars=False,plot_bars_all=False):
    perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'
    if ref:
        hpx_dir_ref='/home/shahrzad/repos/Blazemark/data/matrix/c7/reference/'
        d_hpx_ref=create_dict_reference(hpx_dir_ref)  

    name=''
    descs=[]
    spt_results={}
    for i in range(len(dirs)):
        directory=dirs[i]
        spt_filename='/home/shahrzad/repos/Blazemark/data/data_perf_spt_tmp.csv'
        write_to_file([directory], spt_filename)
        if alias is not None:
            desc=alias[i]
        else:
            desc=directory.split('/')[-3].split('_')[0]+'_'+directory.split('/')[-2]      
        print(desc)
        descs.append(desc)
        name=name+desc+'_'
        spt_results[desc]=create_spt_dict(spt_filename, benchmark)
        node=[k for k in spt_results[desc].keys()][0]
    runtime='hpx'
    b='4-256'
    descs.append('equal')
    
    titles=['runtime','node','benchmark','matrix_size','num_threads','block_size_row','block_size_col','num_elements','work_per_core','w1','w2','w3','w4','w5','w6','w7','w8','chunk_size','grain_size','num_blocks','num_blocks/chunk_size','num_elements*chunk_size','num_blocks/num_threads','num_blocks/(chunk_size*(num_threads-1))','L1cache','L2cache','L3cache','cache_line','set_associativity','datatype','cost','simd_size','execution_time','num_tasks','mflops','include']
    
    dataframe = pandas.read_csv('/home/shahrzad/repos/Blazemark/data/data_perf_all.csv', header=0,index_col=False,dtype=str,names=titles)
    for col in titles[3:]:
        dataframe[col] = dataframe[col].astype(float)    
    nodes=dataframe['node'].drop_duplicates().values
    nodes.sort()
#    node=nodes[0]
    
    benchmark='dmatdmatadd'
    node_selected=dataframe['node']==node
    included=dataframe['include']==1
    df_n_selected=dataframe[node_selected & included]
    benchmark_selected=dataframe['benchmark']==benchmark
    rt_selected=dataframe['runtime']==runtime
    num_threads_selected=dataframe['num_threads']<=8
    df_nb_selected=df_n_selected[benchmark_selected & num_threads_selected & rt_selected]         
    block_selected_r=df_nb_selected['block_size_row']==4
    block_selected_c=df_nb_selected['block_size_col']==256
    df_nb_selected=df_nb_selected[ block_selected_r & block_selected_c]
    thr=df_nb_selected['num_threads'].drop_duplicates().values
    thr.sort()
    matrix_sizes=df_nb_selected['matrix_size'].drop_duplicates().values
    matrix_sizes.sort()
    colors=['red', 'green', 'purple', 'pink', 'cyan', 'lawngreen', 'yellow']
    color_map={}
    for c,desc in zip(colors,['adaptive','adaptive with threshold','guided','guided with threshold']):
        color_map[desc]=c
    color_map['equal']=colors[4]    
    results={}
    results_th={}

    for th in thr:
        results_th[th]={}
        for desc in descs:
            results_th[th][desc]=0
        
    i=1
    for m in [264, 1825]: #matrix_sizes: #
        results[m]={}
        m_selected=df_nb_selected['matrix_size']==m
        features=['chunk_size','num_blocks','num_threads','grain_size','block_size_row','block_size_col','work_per_core','num_tasks','execution_time']
        df_selected=df_nb_selected[m_selected][features]
    
        array_b=df_selected.values
        for th in thr:   
            results[m][th]={}

            new_array=array_b[array_b[:,2]==th][:,:-1]
            new_labels=array_b[array_b[:,2]==th][:,-1]
            results[m][th]['min']=np.min(new_labels)
            results[m][th]['equal']=new_labels[find_nearest_index(new_array[:,3],(m**2)/th)]
            results_th[th]['equal']+=results[m][th]['min']/results[m][th]['equal']

            if plot:
                plt.figure(i)
                plt.axes([0, 0, 1.5, 1.5])
                plt.scatter(new_array[:,3],new_labels,label='true')
           
            for c,desc in zip(colors,spt_results.keys()):
#                plt.figure(i)
                if plot:
                    plt.axhline(spt_results[desc][node][b][benchmark][m][th],color=color_map[desc],label=desc)
                results[m][th][desc]=spt_results[desc][node][b][benchmark][m][th][0]
                results_th[th][desc]+=results[m][th]['min']/results[m][th][desc]

            if ref:
                k=d_hpx_ref['marvin_old'][benchmark][th]['size'].index(m)   
                results[m][th]['ref']=(m**2)/d_hpx_ref['marvin_old'][benchmark][th]['mflops'][k]

                if plot:
                    plt.axhline((m**2)/d_hpx_ref['marvin_old'][benchmark][th]['mflops'][k],color=color_map[desc],label='reference')
            if plot:
                plt.axvline((m**2)/th,color='gray',linestyle='dashed')            
                plt.ylabel('Execution Time($\mu{sec}$)')
                plt.xscale('log')
                plt.legend(bbox_to_anchor=(0.08, 0.98), loc=2, borderaxespad=0.)
                if save:
                    plt.savefig(perf_dir+'/'+save_dir_name+'/blazemark_splittable/'+node+'_spt_'+benchmark+'_'+name+str(int(m))+'_'+str(int(th))+'.png',bbox_inches='tight')
                    plt.close()
                i=i+1
        if plot_bars:  
            k=0
            for desc in [l for l in results[m][1].keys() if l!='min']:
                plt.figure(i)
                plt.axes([0, 0, 1.5, 1.5])
        
                width=0.15
                plt.bar(thr-.25+width*k, [results[m][th]['min']/results[m][th][desc] for th in thr], width,label=desc,color=color_map[desc])
                k=k+1
                plt.xlabel('Number of cores')
                plt.ylabel('Speedup')
                plt.xticks(range(1,9))
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
                plt.savefig(perf_dir+'/'+save_dir_name+'/blazemark_splittable/'+name[:-1]+'/compare/'+node+'_spt_'+benchmark+'_'+str(int(m))+'.png',bbox_inches='tight')
            i=i+1
    for th in thr:
        for desc in descs:
            results_th[th][desc]/=len(matrix_sizes)
            
    if plot_bars_all:
        k=0
        for c,desc in zip(colors,descs):
            plt.figure(i)
            plt.axes([0, 0, 1.5, 1.5])
            width=0.15
            plt.bar(thr-.25+width*k, [results_th[th][desc] for th in thr], width,label=desc,color=color_map[desc])
            k=k+1
            plt.xlabel('Number of cores')
            plt.ylabel('Speedup')
            plt.xticks(range(1,9))
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
            plt.savefig(perf_dir+'/'+save_dir_name+'/blazemark_splittable/'+name[:-1]+'/compare/'+node+'_spt_'+benchmark+'_all.png',bbox_inches='tight')
        i=i+1
    return results,results_th
            
            
def my_model_b(ndata,alpha,gamma,mflop,ts): 
    N=ndata[:,2]
    n_t=ndata[:,-1]
    M=np.minimum(n_t,N) 
    L=np.ceil(n_t/(M))
    w_c=ndata[:,-2]
    ps=mflop
    return alpha*L+ts*(1+(gamma)*(M-1))*(w_c)/ps#+(1)*(d*ps)*np.exp(-((g-ps/N)/(k))**2)#+(1+(gamma)*(M-1))*(w_c)#+(1)*(1/(np.sqrt(2*np.pi)*(d)))*np.exp(-((g-dN)/(ps/N))**2)
