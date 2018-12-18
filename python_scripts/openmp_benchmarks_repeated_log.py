#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:37:23 2018

@author: shahrzad
"""

import glob
import numpy as np
from matplotlib import pyplot as plt
import datetime
now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")


openmp_date_str='12-14-2018-1512'
hpx_date_str='12-18-2018-0946'
hpx_before_date_str='12-10-18-0935'
hpx_dir='/home/shahrzad/repos/Blazemark/data/matrix/'+hpx_date_str
openmp_dir='/home/shahrzad/repos/Blazemark/data/openmp/dmatdmatmult/'+openmp_date_str
perf_directory='/home/shahrzad/repos/Blazemark/data/performance_plots'

from matplotlib.backends.backend_pdf import PdfPages

    
#####################################################
                        #hpx refernce
#####################################################
hpx_before_dir='/home/shahrzad/repos/Blazemark/data/matrix/'+hpx_before_date_str                    
thr=[]
sizes=[]
repeats=[]
block_sizes=[]
chunk_sizes=[]

data_files=glob.glob(hpx_before_dir+'/*.dat')

hpx_before_benchmarks=[]
for filename in data_files:
    if 'hpx' in filename:
        (repeat, benchmark, th, runtime, chunk_size, block_size) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in hpx_before_benchmarks:
            hpx_before_benchmarks.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        if int(block_size) not in block_sizes:
            block_sizes.append(int(block_size))
        if int(chunk_size) not in chunk_sizes:
            chunk_sizes.append(int(chunk_size))
        
thr.sort()
hpx_before_benchmarks.sort()      
repeats.sort()      
block_sizes.sort()
chunk_sizes.sort()

d_hpx_before_all={}   
d_hpx_before={}
for benchmark in hpx_before_benchmarks:  
    d_hpx_before_all[benchmark]={}
    d_hpx_before[benchmark]={}
    for c in chunk_sizes:
        d_hpx_before_all[benchmark][c]={}
        d_hpx_before[benchmark][c]={}
        for b in block_sizes:
            d_hpx_before_all[benchmark][c][b]={}
            d_hpx_before[benchmark][c][b]={}
            for th in thr:
                d_hpx_before_all[benchmark][c][b][th]={}
                d_hpx_before[benchmark][c][b][th]={}
                for r in repeats:
                    d_hpx_before_all[benchmark][c][b][th][r]={}
                
data_files.sort()        
for filename in data_files:    
    if 'hpx' in filename:
        size=[]
        mflops=[]    
        stop=False
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
        chunk=int(filename.split('/')[-1].split('-')[4])       
        block=int(filename.split('/')[-1].split('-')[5][0:-4])       

            
        for r in result:        
            if "N=" in r:
                stop=True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))
        
        d_hpx_before_all[benchmark][chunk][block][th][repeat]['size']=size
        d_hpx_before_all[benchmark][chunk][block][th][repeat]['mflops']=mflops
 
    
for benchmark in hpx_before_benchmarks:
    for c in chunk_sizes:
        for b in block_sizes:
            for th in thr:
                mflops=[0]*len(size)
                d_hpx_before[benchmark][c][b][th]['size']=size

                if max(repeats)==1:
                    mflops=d_hpx_before_all[benchmark][c][b][th][repeats[0]]['mflops']
                    d_hpx_before[benchmark][c][b][th]['mflops']=mflops
                else:
                    for r in repeats[1:]:
                        mflops=[mflops[i]+d_hpx_before_all[benchmark][c][b][th][r]['mflops'][i] for i in range(len(mflops))]                        
                        d_hpx_before[benchmark][c][b][th]['mflops']=[x/float(max(repeats)-1) for x in mflops] 
########################################################################
#hpx
##########################################################################                        
thr=[]
sizes=[]
repeats=[]
block_sizes=[]
chunk_sizes=[]

data_files=glob.glob(hpx_dir+'/*.dat')

hpx_benchmarks=[]
for filename in data_files:
    if 'hpx' in filename:
        (repeat, benchmark, th, runtime, chunk_size, block_size) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in hpx_benchmarks:
            hpx_benchmarks.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        if int(block_size) not in block_sizes:
            block_sizes.append(int(block_size))
        if int(chunk_size) not in chunk_sizes:
            chunk_sizes.append(int(chunk_size))
        
thr.sort()
hpx_benchmarks.sort()      
repeats.sort()      
block_sizes.sort()
chunk_sizes.sort()

thr=[16]
d_hpx_all={}   
d_hpx={}
for benchmark in hpx_benchmarks:  
    d_hpx_all[benchmark]={}
    d_hpx[benchmark]={}
    for c in chunk_sizes:
        d_hpx_all[benchmark][c]={}
        d_hpx[benchmark][c]={}
        for b in block_sizes:
            d_hpx_all[benchmark][c][b]={}
            d_hpx[benchmark][c][b]={}
            for th in thr:
                d_hpx_all[benchmark][c][b][th]={}
                d_hpx[benchmark][c][b][th]={}
                for r in repeats:
                    d_hpx_all[benchmark][c][b][th][r]={}
                
data_files.sort()        
for filename in data_files:    
    if 'hpx' in filename:
        
        stop=False
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
        chunk=int(filename.split('/')[-1].split('-')[4])       
        block=int(filename.split('/')[-1].split('-')[5][0:-4])       

        if block in block_sizes:  
            size=[]
            mflops=[]    
            for r in result:        
                if "N=" in r:
                    stop=True
                if not stop:
                    size.append(int(r.strip().split(' ')[0]))
                    mflops.append(float(r.strip().split(' ')[-1]))
            
            d_hpx_all[benchmark][chunk][block][th][repeat]['size']=size
            d_hpx_all[benchmark][chunk][block][th][repeat]['mflops']=mflops
 
    
for benchmark in hpx_benchmarks:
    for c in chunk_sizes:
        for b in block_sizes:
            for th in thr:
                mflops=[0]*len(size)
                d_hpx[benchmark][c][b][th]['size']=size

                if max(repeats)==1:
                    mflops=d_hpx_all[benchmark][c][b][th][repeats[0]]['mflops']
                    d_hpx[benchmark][c][b][th]['mflops']=mflops
                else:
                    for r in repeats[1:]:
                        mflops=[mflops[i]+d_hpx_all[benchmark][c][b][th][r]['mflops'][i] for i in range(len(mflops))]                        
                        d_hpx[benchmark][c][b][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
       
####################################################
#opemp
####################################################        
data_files=glob.glob(openmp_dir+'/*.dat')
d_openmp={}

openmp_benchmarks=[]
for filename in data_files:
    if 'openmp' in filename:        
        (repeat,benchmark, th, runtime) = filename.split('/')[-1].replace('.dat','').split('-')
        if benchmark not in openmp_benchmarks:
            openmp_benchmarks.append(benchmark)
            
        if int(th) not in thr:
            thr.append(int(th))
        if int(repeat) not in repeats:
            repeats.append(int(repeat))
        
thr.sort()
openmp_benchmarks.sort()      
repeats.sort()      

d_openmp_all={}   

for repeat in repeats:
    d_openmp_all[repeat]={}
    for benchmark in openmp_benchmarks:   
        d_openmp_all[repeat][benchmark]={}
        for th in thr:
            d_openmp_all[repeat][benchmark][th]={}

     

data_files.sort()        
for filename in data_files:    
    size=[]
    mflops=[]    

    if 'openmp' in filename:
        f=open(filename, 'r')
                 
        result=f.readlines()[3:]
        stop=False
        benchmark=filename.split('/')[-1].split('-')[1]
        th=int(filename.split('/')[-1].split('-')[2])       
        repeat=int(filename.split('/')[-1].split('-')[0])       
    
            
        for r in result:
            if "N=" in r:
                stop=True
            if not stop:
                size.append(int(r.strip().split(' ')[0]))
                mflops.append(float(r.strip().split(' ')[-1]))
        
        d_openmp_all[repeat][benchmark][th]['size']=size
        d_openmp_all[repeat][benchmark][th]['mflops']=mflops
 
d_openmp={}
for benchmark in openmp_benchmarks:
    d_openmp[benchmark]={}
    for th in thr:
        d_openmp[benchmark][th]={}
        size_0=d_openmp_all[1][benchmark][th]['size']
        mflops=[0]*len(size_0)
        for r in repeats[1:]:
            size=d_openmp_all[r][benchmark][th]['size']
            if size!=size_0:
                print("errorrrrrrrrrrrrrrrrrrrrr")
            mflops=[mflops[i]+d_openmp_all[r][benchmark][th]['mflops'][i] for i in range(len(mflops))]
        d_openmp[benchmark][th]['size']=size_0
        d_openmp[benchmark][th]['mflops']=[x/float(max(repeats)-1) for x in mflops]
        
        
##############################################################################
        #plots
##############################################################################
pp = PdfPages(perf_directory+'/perf_log_hpx_'+date_str+'.pdf')

i=1
for benchmark in openmp_benchmarks:       
    plt.figure(i)
    for th in thr:
        plt.plot(d_openmp[benchmark][th]['size'], d_openmp[benchmark][th]['mflops'],label=str(th)+' threads')
        plt.xlabel("# matrix size")           
        plt.ylabel('MFlops')
        plt.xscale('log')
        plt.grid(True, 'both')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('openmp   '+benchmark+'  '+date_str)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')
    i=i+1
    
    for c in chunk_sizes:
        for b in block_sizes:     
            plt.figure(i)
            for th in thr:
                plt.plot(d_hpx[benchmark][c][b][th]['size'], d_hpx[benchmark][c][b][th]['mflops'],label='chunk_size: '+str(c)+' block_size: '+str(b)+ '  '+str(th)+' threads')
                plt.xlabel("# matrix size")           
                plt.ylabel('MFlops')
                plt.xscale('log')
                plt.grid(True, 'both')
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.title('hpx   '+benchmark+'  '+date_str)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            i=i+1    
            plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')
    
    plt.figure(i)
    for th in thr:
        plt.plot(d_hpx_before[benchmark][10][256][th]['size'], d_hpx_before[benchmark][10][256][th]['mflops'],label=str(th)+' threads')
        plt.xlabel("# matrix size")           
        plt.ylabel('MFlops')
        plt.xscale('log')
        plt.grid(True, 'both')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('hpx   before changes '+benchmark+'  '+date_str)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    print('')
    i=i+1
        
plt.show()
pp.close() 
        
import pickle
#################################
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
##################################       
        
save_obj(d_openmp, openmp_dir+'/openmp_benchmarks')
