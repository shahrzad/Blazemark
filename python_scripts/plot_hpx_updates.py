#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:35:07 2019

@author: shahrzad
"""

import glob
import numpy as np
from matplotlib import pyplot as plt
import datetime
now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")


#hpx_dir='/home/shahrzad/repos/Blazemark/data/matrix/dmatdmatadd/01-04-2019-1027'
hpx_dir='/home/shahrzad/repos/Blazemark/data/matrix/dmatdmatmult/hpx_chunk_size/chunk_commented/2000'
openmp_dir_2='/home/shahrzad/repos/Blazemark/data/openmp/04-27-2019/'
openmp_dir_1='/home/shahrzad/repos/Blazemark/data/openmp/all/'


perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/dmatdmatadd/hpx/06-13-2019'

from matplotlib.backends.backend_pdf import PdfPages

def create_dict(directory):
    thr=[]
    repeats=[]
    data_files=glob.glob(directory+'/*.dat')
    benchmark=''
    benchmarks=[]
    chunk_sizes=[]
    block_sizes=[]
    for filename in data_files:
        (repeat, benchmark, th, runtime, chunk_size, block_size_row, block_size_col) = filename.split('/')[-1].replace('.dat','').split('-')         
        if benchmark not in benchmarks:
                benchmarks.append(benchmark)                
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat)) 
        if int(chunk_size) not in chunk_sizes:
            chunk_sizes.append(int(chunk_size))
        if block_size_row+'-'+block_size_col not in block_sizes:
            block_sizes.append(block_size_row+'-'+block_size_col)
                  
    thr.sort()
    benchmarks.sort()      
    repeats.sort()      
    chunk_sizes.sort()
    block_sizes.sort()
    mat_sizes={}
    
    d_all={}   
    d={}
    for benchmark in benchmarks:  
        d_all[benchmark]={}
        d[benchmark]={}
        for th in thr:
            d_all[benchmark][th]={}
            d[benchmark][th]={}
            for r in repeats:
                d_all[benchmark][th][r]={}        
                for bs in block_sizes:
                    d_all[benchmark][th][r][bs]={}
                    d[benchmark][th][bs]={}
                    for cs in chunk_sizes:
                        d_all[benchmark][th][r][bs][cs]={}
                        d[benchmark][th][bs][cs]={}

                                            
    data_files.sort()        
    for filename in data_files:                
        stop=False
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])  
        chunk_size=int(filename.split('/')[-1].split('-')[4]) 
        block_size=filename.split('/')[-1].split('-')[5]+'-'+filename.split('/')[-1].split('-')[6][0:-4]
        size=[]
        mflops=[]    
        for r in result:        
            if "N=" in r:
                stop=True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))
            
        d_all[benchmark][th][repeat][block_size][chunk_size]['size']=size
        d_all[benchmark][th][repeat][block_size][chunk_size]['mflops']=mflops
        if 'size' not in d[benchmark][th][block_size][chunk_size].keys():
            d[benchmark][th][block_size][chunk_size]['size']=size
            d[benchmark][th][block_size][chunk_size]['mflops']=[0]*len(size)
        if len(repeats)==1 and repeat==1:
            d[benchmark][th][block_size][chunk_size]['mflops']=mflops
        elif len(repeats)>1 and repeat!=1:
            d[benchmark][th][block_size][chunk_size]['mflops']+=mflops/(len(repeats)-1)
        else:
            print("errrrrorrrrrrrrrrrr")
        if benchmark not in mat_sizes.keys():
            mat_sizes[benchmark]=size
    return (d, chunk_sizes, block_sizes, thr, benchmarks, mat_sizes)      

#######################################################
def create_dict_relative(directory):
    thr=[]
    repeats=[]
    data_files=glob.glob(directory+'/*.dat')
    benchmark=''
    benchmarks=[]
    chunk_sizes=[]
    block_sizes=[]
    mat_sizes={}
    
    for filename in data_files:
        (repeat, benchmark, th, runtime, chunk_size, block_size_row, block_size_col, mat_size) = filename.split('/')[-1].replace('.dat','').split('-')         
        if benchmark not in benchmarks:
                benchmarks.append(benchmark)   
                mat_sizes[benchmark]=[]
        if int(mat_size) not in mat_sizes[benchmark]:
            mat_sizes[benchmark].append(int(mat_size))
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat)) 
        if int(chunk_size) not in chunk_sizes:
            chunk_sizes.append(int(chunk_size))
        if block_size_row+'-'+block_size_col not in block_sizes:
            block_sizes.append(block_size_row+'-'+block_size_col)
                  
    thr.sort()
    benchmarks.sort()      
    repeats.sort()      
    chunk_sizes.sort()
    block_sizes.sort()
    
    
    d_all={}   
    d={}
    for benchmark in benchmarks:  
        mat_sizes[benchmark].sort()
        d_all[benchmark]={}
        d[benchmark]={}
        for th in thr:
            d_all[benchmark][th]={}
            d[benchmark][th]={}
            for r in repeats:
                d_all[benchmark][th][r]={}        
                for bs in block_sizes:
                    d_all[benchmark][th][r][bs]={}
                    d[benchmark][th][bs]={}
                    for cs in chunk_sizes:
                        d_all[benchmark][th][r][bs][cs]={}
                        d[benchmark][th][bs][cs]={}
                        d[benchmark][th][bs][cs]['size']=mat_sizes[benchmark]
                        d_all[benchmark][th][r][bs][cs]['size']=mat_sizes[benchmark]
                        d_all[benchmark][th][r][bs][cs]['mflops']=[0]*len(mat_sizes[benchmark])

                                            
    data_files.sort()        
    for filename in data_files:                
        stop=False
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])  
        chunk_size=int(filename.split('/')[-1].split('-')[4]) 
        block_size=filename.split('/')[-1].split('-')[5]+'-'+filename.split('/')[-1].split('-')[6].replace('.dat','')    
        for r in result:        
            if "N=" in r:
                stop=True
            if not stop:
                s=mat_sizes[benchmark].index(int(r.strip().split(' ')[0]))
                d_all[benchmark][th][repeat][block_size][chunk_size]['mflops'][s]=float(r.strip().split(' ')[-1])

#        if 'size' not in d[benchmark][th][block_size][chunk_size].keys():
#            d[benchmark][th][block_size][chunk_size]['size']=size
#            d[benchmark][th][block_size][chunk_size]['mflops']=[0]*len(size)
        if len(repeats)==1 and repeat==1:
            d[benchmark][th][block_size][chunk_size]['mflops']=d_all[benchmark][th][repeat][block_size][chunk_size]['mflops']
        elif len(repeats)>1 and repeat!=1:
            d[benchmark][th][block_size][chunk_size]['mflops']+=d_all[benchmark][th][repeat][block_size][chunk_size]['mflops']/(len(repeats)-1)
        else:
            print("errrrrorrrrrrrrrrrr")

    return (d, chunk_sizes, block_sizes, thr, benchmarks, mat_sizes)  
###########################################################################
    
def create_dict_openmp(directory):
    thr=[]
    repeats=[]
    data_files=glob.glob(directory+'/*.dat')
    benchmark=''
    benchmarks=[]
    for filename in data_files:
        (repeat, benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')         
        if benchmark not in benchmarks:
                benchmarks.append(benchmark)                
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))          
                  
    thr.sort()
    benchmarks.sort()      
    repeats.sort()      
    
    d_all={}   
    d={}
    for benchmark in benchmarks:  
        d_all[benchmark]={}
        d[benchmark]={}
        for th in thr:
            d_all[benchmark][th]={}
            d[benchmark][th]={}
            for r in repeats:
                d_all[benchmark][th][r]={}        
                                            
    data_files.sort()        
    for filename in data_files:                
        stop=False
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])        
        size=[]
        mflops=[]    
        for r in result:        
            if "N=" in r:
                stop=True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))
            
        d_all[benchmark][th][repeat]['size']=size
        d_all[benchmark][th][repeat]['mflops']=mflops
 
        
    for benchmark in benchmarks:
        for th in thr:
            d[benchmark][th]['size']=d_all[benchmark][th][1]['size']
            mflops=[0]*len(d[benchmark][th]['size'])    
            if max(repeats)==1:
                if 'mflops' in d_all[benchmark][th][1].keys():
                    mflops=d_all[benchmark][th][repeats[0]]['mflops']
                    d[benchmark][th]['mflops']=mflops
            else:
                for r in repeats[1:]:
                    mflops=[mflops[i]+d_all[benchmark][th][r]['mflops'][i] for i in range(len(mflops))]                        
                    d[benchmark][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
    return d                           
           
(d_hpx,  chunk_sizes, block_sizes, thr, benchmarks, mat_sizes)=create_dict(hpx_dir)                 
d_openmp=create_dict_openmp(openmp_dir_1)

i=1
for benchmark in benchmarks:
    for th in thr:
        plt.figure(i)
        for a in ['4']:        
            for b in block_sizes:
                if b.startswith(a+'-'):
                    for c in chunk_sizes:   
                        
                        if d_hpx[benchmark][th][b][c]['mflops'].count(0)<0.5*len(d_hpx[benchmark][th][b][c]['mflops']):
                            plt.plot(d_hpx[benchmark][th][b][c]['size'], d_hpx[benchmark][th][b][c]['mflops'],label='chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads',marker='*')
    #                        plt.plot(d_openmp[benchmark][th]['size'], d_openmp[benchmark][th]['mflops'],label='openmp '+str(th)+' threads')
    
                            plt.xlabel("# matrix size")           
                            plt.ylabel('MFlops')
                            plt.xscale('log')
                            plt.grid(True, 'both')
                            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                            plt.title('hpx   '+benchmark)
                        i=i+1    
                        plt.figure(i)
#                            plt.savefig(pp, format='pdf',bbox_inches='tight')
                        print('')        
#plt.show()
#pp.close()                   

import math
b='4-1024'
b_r=int(b.split('-')[0])
b_c=int(b.split('-')[1])
BR=[]
BC=[]
c=10
th=16
simdsize=4
d={}
i=0
for benchmark in benchmarks:
    for b in block_sizes:
        num_blocks=[]
        plt.figure(i)
        d[b]={}
        j=0
        for m in mat_sizes[benchmark]: 
            b_r=int(b.split('-')[0])
            b_c=int(b.split('-')[1])
            rest2=m%simdsize
            if b_r>m:
                b_r=m
            if b_c>m:
                b_c=m
            equalshare1=math.ceil(m/b_r)
            equalshare2=math.ceil(m/b_c+rest2)  
            if 'mflops' in d_hpx[benchmark][th][b][c].keys():
                d[b][m]=[b_r,b_c,d_hpx[benchmark][th][b][c]['mflops'][j],equalshare1*equalshare2]
                j=j+1
                BR.append(b_r)
                BC.append(b_c)
                num_blocks.append(equalshare1*equalshare2)

            plt.scatter(num_blocks,d[benchmark][th][b][c]['mflops'])
            plt.xlabel("# blocks")           
            plt.ylabel('MFlops')
            plt.xscale('log')
            plt.grid(True, 'both')
            i=i+1

X=np.zeros((len(block_sizes),60))
Y=np.zeros((len(block_sizes),60))
Z=np.zeros((len(block_sizes),60))
all_points={}
points=np.zeros((len(block_sizes)*60,3))

j=0
for b in block_sizes:
    for i in range(60):
        m=mat_sizes[benchmark][i]
        points[60*j+i][0]=np.log10(m)
        points[60*j+i][1]=d[b][m][3]
        points[60*j+i][2]=d[b][m][2]
        X[j,i]=m
        Y[j,i]=d[b][m][3]
        Z[j,i]=d[b][m][2]
    j=j+1
    all_points[b]=points
    
for k in range(50,59):
    b=Z[:,k]
    a=Y[:,k]
    p=np.argsort(a)
    b[p]
    plt.figure(i)
    plt.plot(a[p], b[p],label=str(int(X[0,k])))
    plt.xscale('log')
    plt.legend()
    i=i+1
    
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y, X, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


import csv
f=open('/home/shahrzad/repos/Blazemark/data/data.csv','w')
f_writer=csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
f_writer.writerow(['benchmark','matrix_size','num_threads','block_size','chunk_size','mflops'])
for benchmark in d_hpx.keys():
    for th in d_hpx[benchmark].keys():
        for block_size in d_hpx[benchmark][th].keys():
            for chunk_size in d_hpx[benchmark][th][block_size].keys():
                if len(d_hpx[benchmark][th][block_size][chunk_size])!=0:
                    for i in range(len(d_hpx[benchmark][th][block_size][chunk_size]['size'])):
                        f_writer.writerow([benchmark,str(d_hpx[benchmark][th][block_size][chunk_size]['size'][i]),str(th),block_size,str(chunk_size),str(d_hpx[benchmark][th][block_size][chunk_size]['mflops'][i])])
f.close()                    
  
(d_hpx_old,  chunk_sizes, block_sizes, thr, benchmarks, mat_sizes_2)=create_dict('/home/shahrzad/repos/Blazemark/results/previous')                 
(d_hpx,  chunk_sizes, block_sizes, thr, benchmarks, mat_sizes)=create_dict(hpx_dir)   
hpx_dir='/home/shahrzad/repos/Blazemark/data/matrix/dmatdmatadd/06-13-2019'              
(d_hpx,  chunk_sizes, block_sizes, thr, benchmarks, mat_sizes)=create_dict_relative(hpx_dir)                 


perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots/dmatdmatadd/hpx/06-13-2019'


i=1
benchmark='dmatdmatadd'
for th in d_hpx[benchmark].keys():
    for b in d_hpx[benchmark][th].keys():
        plt.figure(i)
        pp = PdfPages(perf_directory+'/bath_tub_'+str(th)+'_'+str(b)+'.pdf')
        for m in mat_sizes[benchmark]: 
        
            
            results=[]
            chunk_sizes=[]
            for c in d_hpx[benchmark][th][b]:                    
                k=d_hpx[benchmark][th][b][c]['size'].index(m)
                if 'mflops' in d_hpx[benchmark][th][b][c].keys() and d_hpx[benchmark][th][b][c]['mflops'][k]:
                    chunk_sizes.append(c)
                    results.append(d_hpx[benchmark][th][b][c]['mflops'][k])
            if len(chunk_sizes)!=0:
                b_r=int(b.split('-')[0])
                b_c=int(b.split('-')[1])
                rest1=b_r%simdsize
                rest2=b_c%simdsize
                if b_r>m:
                    b_r=m
                if b_c>m:
                    b_c=m
                equalshare1=math.ceil(m/b_r)
                equalshare2=math.ceil(m/b_c)  
                plt.plot(chunk_sizes, results, label=str(th)+' threads  matrix_size:'+str(m)+'  block_size:'+str(b)+'  num_blocks:'+str(equalshare1*equalshare2))
                plt.xlabel("chunk_size")           
                plt.ylabel('MFlops')
                plt.xscale('log')
                plt.grid(True, 'both')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                print('')     
                plt.savefig(pp, format='pdf',bbox_inches='tight')
            i=i+1
            
        plt.show()
        pp.close() 

i=1
benchmark='dmatdmatadd'
for th in d_hpx[benchmark].keys():
    for b in d_hpx[benchmark][th].keys():
        plt.figure(i)
        pp = PdfPages(perf_directory+'/bath_tub_'+str(th)+'_'+str(b)+'.pdf')
        for m in mat_sizes[benchmark]: 
        
            
            results=[]
            chunk_sizes=[]
            for c in d_hpx[benchmark][th][b]:                    
                k=d_hpx[benchmark][th][b][c]['size'].index(m)
                if 'mflops' in d_hpx[benchmark][th][b][c].keys() and d_hpx[benchmark][th][b][c]['mflops'][k]:
                    chunk_sizes.append(c)
                    results.append(d_hpx[benchmark][th][b][c]['mflops'][k])
            if len(chunk_sizes)!=0:
                b_r=int(b.split('-')[0])
                b_c=int(b.split('-')[1])
                rest1=b_r%simdsize
                rest2=b_c%simdsize
                if b_r>m:
                    b_r=m
                if b_c>m:
                    b_c=m
                equalshare1=math.ceil(m/b_r)
                equalshare2=math.ceil(m/b_c)  
                plt.plot(chunk_sizes, results, label=str(th)+' threads  matrix_size:'+str(m)+'  block_size:'+str(b)+'  num_blocks:'+str(equalshare1*equalshare2))
                plt.xlabel("chunk_size")           
                plt.ylabel('MFlops')
                plt.xscale('log')
                plt.grid(True, 'both')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                print('')     
                plt.savefig(pp, format='pdf',bbox_inches='tight')
            i=i+1
            
        plt.show()
        pp.close() 
        
for th in d_hpx[benchmark].keys():
    pp = PdfPages(perf_directory+'/'+str(th)+'.pdf')

    for m in mat_sizes[benchmark]:

        plt.figure(i)
        for b in d_hpx[benchmark][th].keys():
            results=[]
            chunk_sizes=[]
            for c in d_hpx[benchmark][th][b]:                    
                k=d_hpx[benchmark][th][b][c]['size'].index(m)
                if 'mflops' in d_hpx[benchmark][th][b][c].keys() and d_hpx[benchmark][th][b][c]['mflops'][k]:
                    chunk_sizes.append(c)
                    results.append(d_hpx[benchmark][th][b][c]['mflops'][k])
            if len(chunk_sizes)!=0:
                b_r=int(b.split('-')[0])
                b_c=int(b.split('-')[1])
                rest1=b_r%simdsize
                rest2=b_c%simdsize
                if b_r>m:
                    b_r=m
                if b_c>m:
                    b_c=m
                equalshare1=math.ceil(m/b_r)
                equalshare2=math.ceil(m/b_c)  
                plt.figure(i)
                plt.plot(chunk_sizes, results, label=str(th)+' threads matrix_size:'+str(m)+' block_size:'+str(b)+' num_blocks:'+str(equalshare1*equalshare2))
                plt.xlabel("chunk_size")           
                plt.ylabel('MFlops')
                plt.xscale('log')
                plt.grid(True, 'both')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('') 
        i=i+1
                    
    plt.show()
    pp.close()                  
    
      