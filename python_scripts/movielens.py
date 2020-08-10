#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:20:00 2018

@author: shahrzad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tag_headers = ['user_id', 'movie_id', 'tag', 'timestamp']
tags = pd.read_table('tags.csv', sep=',', header=0, names=tag_headers)

rating_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('ratings.csv', sep=',', header=0, names=rating_headers)
all_r=ratings.pivot_table(index='userId',columns='title',values='rating')

R=ratings.values
all_users=np.unique(R[:,0]).tolist()
all_movies=np.unique(R[:,1]).tolist()

final=np.zeros((110000,len(all_movies)))
for i in range(np.shape(R)[0]):
    print(i)
    final[all_users.index(R[i,0]),all_movies.index(R[i,1])]=R[i,2]

F=final[0:all_users.index(R[i-1,0])]

np.savetxt("MovieLens_"+str(all_users.index(R[i-1,0]))+"_"+str(len(all_movies))+".csv",F,delimiter=',')
    
movie_headers = ['movie_id', 'title', 'genres']
movies = pd.read_table('movies.csv',
                       sep=',', header=0, names=movie_headers)
movie_titles = movies.title.tolist()

df = movies.join(ratings, on=['movie_id'], rsuffix='_r').join(tags, on=['movie_id'], rsuffix='_t')
del df['movie_id_r']
del df['user_id_t']
del df['movie_id_t']
del df['timestamp_t']

rp = df.pivot_table(columns=['movie_id'],index=['user_id'],values='rating')
rp = rp.fillna(0); # Replace NaN
Q = rp.values

#np.save("MovieLens.npy", Q)
#r=np.load("MovieLens.npy")
np.savetxt("MovieLens.csv",Q,delimiter=',')