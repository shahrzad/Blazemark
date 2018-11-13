#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:08:05 2018

@author: shahrzad
"""

import glob
import numpy as np
from matplotlib import pyplot as plt
import datetime
now = datetime.datetime.now()
date_str=now.strftime("%Y-%m-%d-%H%M")
import math
from matplotlib.backends.backend_pdf import PdfPages



thr=np.arange(1,17).tolist()
chunks=[]
blocks=[]

directory='/home/shahrzad/Spyder/Blazemark/results/data/dynamic/log/'
dynamic_date_str='11-08-18-1330'


thr=[]    
block_sizes=[]
chunk_sizes=[]
benchmarks=[]

data_files=glob.glob(directory+'/*.dat')
for filename in data_files:
    b=filename.split('-')[0].split('/')[-1]
    th=filename.split('-')[1]
    chunk=filename.split('-')[3]
    block=filename.split('-')[4]
    if b not in benchmarks:
        benchmarks.append(b)
    if th not in thr:
        thr.append(th)
    if chunk not in chunks:
        chunks.append(chunk)
    if block not in block_sizes:
        block_sizes.append(block) 

thr.sort()
block_sizes.sort()
chunk_sizes.sort()
benchmarks.sort()

d_all={}
for benchmark in benchmarks:
    d_all[benchmark]={}
    for b in block_sizes:
        d_all[benchmark][b]={}
        for th in thr:
            d_all[benchmark][b][str(th)]={}
    
for filename in data_files:
    size=[]
    mflops=[]
    f=open(filename, 'r')
    result=f.readlines()[3:163]
    benchmark=filename.split('-')[0].split('/')[-1]

    th=filename.split('-')[1]
    chunk=filename.split('-')[3]
    block=filename.split('-')[4]
    
   
        
    for r in result:
        size.append(int(r.strip().split(' ')[0]))
        mflops.append(float(r.strip().split(' ')[-1]))
    
    d_all[benchmark][block][th]['size']=size
    d_all[benchmark][block][th]['mflops']=mflops



pp = PdfPages(directory+'/perf_log_'+date_str+'.pdf')

i=1
for benchmark in benchmarks:
    for b in block_sizes:
        for th in thr:
            plt.figure(i)
        #    plt.title('vector size:'+str(v))
            print(th)
            plt.title(benchmark+' block size:'+str(b))
            plt.plot(d_all[benchmark][block][str(th)]['size'], d_all[benchmark][block][str(th)]['mflops'],label=str(th)+' threads')
            plt.xlabel("# vector size")
            plt.ylabel('MFlops')
            plt.xscale('log')
            plt.grid(True, 'both')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        i=i+1
        plt.savefig(pp, format='pdf',bbox_inches='tight')
        print('')
   
plt.show()
pp.close()