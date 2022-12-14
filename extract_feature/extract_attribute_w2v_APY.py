# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:43:05 2019

@author: badat
"""
import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import pdb
import pandas as pd
import numpy as np
import gensim.downloader as api
import pickle
#%%
print('Loading pretrain w2v model')
model_name = 'word2vec-google-news-300'#best model
model = api.load(model_name)
dim_w2v = 300
print('Done loading model')
#%%
replace_word = [('spatulate','broad'),('upperparts','upper parts'),('grey','gray')]
#%%
path = './attribute/APY/attribute_names.txt'
df=pd.read_csv(path,header = None, names = ['des'])
print(df)
des = df['des'].values
new_des=des
#%% filter
# new_des = [' '.join(i.split('_')) for i in des]
# new_des = [' '.join(i.split('-')) for i in new_des]
# new_des = [' '.join(i.split('::')) for i in new_des]
# new_des = [i.split('(')[0] for i in new_des]
# new_des = [i[4:] for i in new_des]
#%% replace out of dictionary words
for pair in replace_word:
    for idx,s in enumerate(new_des):
        new_des[idx]=s.replace(pair[0],pair[1])
print('Done replace OOD words')
#%%
df['new_des']=new_des
df.to_csv('./attribute/APY/new_des.csv')
print('Done preprocessing attribute des')
#%%
all_w2v = []
for s in new_des:
    print(s)
    words = s.split(' ')
    if words[-1] == '':     #remove empty element
        words = words[:-1]
    w2v = np.zeros(dim_w2v)
    for w in words:
        try:
            w2v += model[w]
        except Exception as e:
            print(e)
    all_w2v.append(w2v[np.newaxis,:])
#%%
all_w2v=np.concatenate(all_w2v,axis=0)
pdb.set_trace()
#%%
with open('./w2v/APY_attribute.pkl','wb') as f:
    pickle.dump(all_w2v,f)