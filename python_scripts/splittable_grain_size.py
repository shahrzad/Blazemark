c#!/usr/bin/env python3
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

from scipy.optimize import nnls
#from sklearn.metrics import r2_score

import grain_size_funcs as gf


dir1='/home/shahrzad/repos/Blazemark/data/final/grain_size/marvin/splittable/all/'
dir2='/home/shahrzad/repos/Blazemark/data/final/grain_size/marvin/splittable/idle/'
dir3='/home/shahrzad/repos/Blazemark/data/final/grain_size/marvin/general/'

dir1='/home/shahrzad/repos/Blazemark/data/final/grain_size/medusa/splittable/all/'
dir2='/home/shahrzad/repos/Blazemark/data/final/grain_size/medusa/splittable/idle/'
dir3='/home/shahrzad/repos/Blazemark/data/final/grain_size/medusa/general/'

gf.create_dict([dir3])
dirs=[dir1,dir2]
alias=['guided','adaptive']
save_dir_name='thesis'
gf.compare_results(dirs, save_dir_name, alias, save=1, mode='ps-th')

#dir2='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/2_rem_gt_0/all_cores/'                        
#dir3='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/3_rem_gt_0_mult4/all_cores/'                        
#dir4='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/4_rem_gt_0_mult2/all_cores/'        
dir1='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/after_latch_update/'
dir2='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/5_master_Hartmut/idle_updated/'  
dir3='/home/shahrzad/repos/Blazemark/data/grain_size/marvin/splittable/after_latch_update/'
dir4='/home/shahrzad/repos/Blazemark/data/grain_size/marvin/splittable/before_latch_update/'

dir3='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/5_master_Hartmut/all_2/'  
dir4='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/5_master_Hartmut/all_multiple_tasks_1/'    
dir5='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/6_master_Hartmut_200/all_multiple_tasks_1/'        
dir6='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/7_master_Hartmut_400/all_multiple_tasks_1/'       
dir7='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/8_master_Hartmut_600/all_multiple_tasks_1/'                
#dir7='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/9_master_Hartmut_800/idle_mask_1/'                

#dir3='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/hpx_master/all_cores/'
dirs=[dir2,dir3,dir4,dir5,dir6,dir7]
alias=['idle_mask','all','all_multiple'],'all_multiple_200','all_multiple_400', 'all_multiple_600']
alias=['idle_mask','all','all_multiple']
alias=['idle_mask','all'],'all_2']

dir5='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/5_master_Hartmut/idle_updated_qs/'        
dir6='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/5_master_Hartmut/idle_updated/'        

dirs=[dir2,dir3,dir4,dir5,dir6]
alias=['idle_mask','all','all_multiple','idle_updated_qs','idle_updated']
save_dir_name='5-6-7-8/idle_updated_qs/compare_qs_only/'

dirs=[dir2,dir3],dir4]
#alias=['idle_mask','idle_mask_200','idle_mask_400', 'idle_mask_600', 'idle_mask_800']
save_dir_name='5-6-7-8/idle_mask_all_1/ps/'
#gf.compare_results([dir2,dir4,dir3], save_dir_name,['all_cores*1','all_cores*2','all_cores*4'])
#gf.compare_results(dirs, save_dir_name,alias,1)
save_dir_name='5-6-7-8/compare_th/all_multiple_tasks_1/'
save_dir_name='5-6-7-8/compare_th/different_modes/'
save_dir_name='5-6-7-8/compare_th/ps/'

dirs=[dir5,dir6]
alias=['idle_mask_iter_1', 'idle_mask_iter_1000']
iteration_lengths=[1]
gf.compare_results(dirs, save_dir_name, alias, save=1, mode='ps-th')
gf.compare_results(dirs, save_dir_name, alias, save=1, mode='ps')
gf.compare_results(dirs, save_dir_name, alias, save=1, mode='th')

dirs=[dir1,dir2]
gf.compare_results(dirs, 'latch_update/c7/', ['idle-latch_updated','idle'], save=1, mode='ps-th')
gf.compare_results([dir3,dir4], 'latch_update/marvin/', ['idle-latch_updated','idle'], save=1, mode='ps-th')

gf.compare_results([dir1,dir2,dir3,dir4], 'latch_update/', ['idle-latch_updated','idle','idle-latch_updated_marvin','idle_marvin'], save=0, mode='ps-th')

perf_counter_dirs=[dir2,dir3,dir4]
save_dir_name='5-6-7-8/perf_counters/idle_mask_1/exec/'
perf_dict=gf.get_task_info(perf_counter_dirs, save_dir_name, plot=1, save=1)

dir2='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/5_master_Hartmut/idle_mask_perf_1/'  
perf_counter_dirs=[dir2]
save_dir_name='5-6-7-8/perf_counters/idle_mask_1/'
perf_dict=gf.get_task_info(perf_counter_dirs, save_dir_name, plot=1, save=1, value_sorted=1, plot_reps=1)
#dir2='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/2_rem_gt_0/idle_cores/'                        
#dir3='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/3_rem_gt_0_mult4/idle_cores/'                        
#dir4='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/4_rem_gt_0_mult2/idle_cores/'                        
#
#save_dir_name='2-3-4/idle_cores'
#gf.compare_results([dir2,dir4,dir3], save_dir_name,['idle_cores*1','idle_cores*2','idle_cores*4'])

marvin_dir='/home/shahrzad/repos/Blazemark/data/grain_size/marvin'
medusa_dir='/home/shahrzad/repos/Blazemark/data/grain_size/medusa'
tameshk_dir='/home/shahrzad/repos/Blazemark/data/grain_size/tameshk'
marvin_old_dir_idle='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/idle_cores/2/'
marvin_old_dir_all_2='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/2_stop_start/all_cores/'
marvin_old_dir_all_3='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/3_stop_start_mult4/all_cores/'

perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'

#marvin_old_dir_all_master='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/hpx_master/all_cores/'
#marvin_old_dir_idle_master='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/hpx_master/idle_cores/'

#filename_idle_master='/home/shahrzad/repos/Blazemark/data/grain_data_perf_master_idle_cores.csv'
#filename_all_master='/home/shahrzad/repos/Blazemark/data/grain_data_perf_master_all_cores.csv'
#gf.create_dict([marvin_old_dir_idle_master],data_filename=filename_idle_master)
#gf.create_dict([marvin_old_dir_all_master],data_filename=filename_all_master)
#
#spt_results_all_master=gf.create_spt_dict(filename_all_master)
#spt_results_idle_master=gf.create_spt_dict(filename_idle_master)

filename_idle='/home/shahrzad/repos/Blazemark/data/grain_data_perf_idle_cores.csv'
filename_idle_min='/home/shahrzad/repos/Blazemark/data/grain_data_perf_idle_cores_min.csv'
filename_idle_max='/home/shahrzad/repos/Blazemark/data/grain_data_perf_idle_cores_max.csv'
filename_all_3='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all_cores_3.csv'
filename_all_2='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all_cores_2.csv'

gf.create_dict([marvin_old_dir_idle],data_filename=filename_idle)
gf.create_dict([marvin_old_dir_idle],data_filename=filename_idle_min,mini=True)
gf.create_dict([marvin_old_dir_idle],data_filename=filename_idle_max,maxi=True)
gf.create_dict([marvin_old_dir_all_2],data_filename=filename_all_2)
gf.create_dict([marvin_old_dir_all_3],data_filename=filename_all_3)



spt_results_all_2=gf.create_spt_dict(filename_all_2)
spt_results_all_3=gf.create_spt_dict(filename_all_3)

spt_results_idle=gf.create_spt_dict(filename_idle)
spt_results_idle_min=gf.create_spt_dict(filename_idle_min)
spt_results_idle_max=gf.create_spt_dict(filename_idle_max)


node='marvin'
threads={}

titles=['node','problem_size','num_blocks','num_threads','chunk_size','iter_length','grain_size','work_per_core','num_tasks','execution_time']
filename='/home/shahrzad/repos/Blazemark/data/grain_data_perf_all.csv'
perf_dir='/home/shahrzad/repos/Blazemark/data/performance_plots/06-13-2019/hpx_for_loop/general'

dataframe = pandas.read_csv(filename, header=0,index_col=False,dtype=str,names=titles)
for col in titles[1:]:
    dataframe[col] = dataframe[col].astype(float)
    
node_selected=dataframe['node']==node
nt_selected=dataframe['num_tasks']>=1
iter_selected=dataframe['iter_length']==1
th_selected=dataframe['num_threads']>=1
df_n_selected=dataframe[node_selected & nt_selected & iter_selected & th_selected][titles[1:]]

problem_sizes=df_n_selected['problem_size'].drop_duplicates().values
problem_sizes.sort()
thr=df_n_selected['num_threads'].drop_duplicates().values
thr.sort()
threads[node]=thr

i=1
array=df_n_selected.values
for ps in spt_results_all_2['c7-spt'].keys():
    array_ps=array[array[:,0]==ps]
    if np.shape(array_ps)[0]!=0:
        for th in thr:
            plt.figure(i)
            array_t=array_ps[array_ps[:,2]==th]
            plt.scatter(array_t[:,5],array_t[:,-1])
            plt.axhline(spt_results_all_2['c7-spt'][ps][th], color='green', label='all cores')
#            plt.axhline(spt_results_idle['c7-spt'][ps][th], color='red', label='idle cores-avg',linestyle='dashed')
            plt.axhline(spt_results_all_3['c7-spt'][ps][th], color='purple', label='all cores - 4*threads')

#            plt.axhline(spt_results_idle_min['c7-spt'][ps][th], color='red', label='idle cores-min',linestyle='dashed')
#            plt.axhline(spt_results_idle_max['c7-spt'][ps][th], color='orange', label='idle cores-max',linestyle='dashed')

#            plt.axhline(spt_results_all_master['c7-spt'][ps][th], color='green', alpha=0.5,label='all cores-master', linestyle='dashed')
#            plt.axhline(spt_results_idle_master['c7-spt'][ps][th], color='red', alpha=0.5,label='idle cores-master', linestyle='dashed')

#            plt.axvline(ps/th,color='gray',linestyle='dashed')
            plt.xlabel('Grain size')
            plt.ylabel('Execution time')
            plt.xscale('log')
            plt.title('ps='+str(int(ps))+' '+str(int(th))+' threads')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            i=i+1
            plt.savefig(perf_dir+'/splittable/3-2/all_cores/'+node+'_'+str(int(ps))+'_'+str(int(th))+'.png',bbox_inches='tight')
#            plt.savefig(perf_dir+'/splittable/hpx_master/master_old/all_idle_ref/'+node+'_'+str(int(ps))+'_'+str(int(th))+'.png',bbox_inches='tight')


marvin_old_dir_split_idle='/home/shahrzad/repos/Blazemark/data/grain_size/c7/splittable/idle_cores/split_info/2/'
