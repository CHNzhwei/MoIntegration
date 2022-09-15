#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:29:36 2019

@author: wanxiang.shen@u.nus.edu

main molmap code

"""

from utils.Extraction import Extraction
from utils.logtools import print_info, print_warn, print_error
from utils.matrixopt import Scatter2Grid, Scatter2Array 

from utils import vismap


from sklearn.manifold import TSNE, MDS
from sklearn.utils import shuffle
from joblib import Parallel, delayed, load, dump
from umap import UMAP
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import h5py



class Base:
    
    def __init__(self):
        pass
        
    def _save(self, filename):
        return dump(self, filename)
        
    def _load(self, filename):
        return load(filename)
 

    def MinMaxScaleClip(self, x, xmin, xmax):
        scaled = (x - xmin) / ((xmax - xmin) + 1e-8)
        return scaled.clip(0, 1)

    def StandardScaler(self, x, xmean, xstd):
        return (x-xmean) / (xstd + 1e-8) 
    
    
    

class omicsMap(Base):
    
    def __init__(self, 
                 fmap_type, 
                 fmap_shape, 
                 split_channels,
                 metric, 
                 path,
                 omics_colormaps):
        """
        paramters
        -----------------
        flist: feature list, if you want use some of the features instead of all features, each element in flist should be the id of a feature
        fmap_shape: None or tuple, size of molmap, only works when fmap_type is 'scatter', if None, the size of feature map will be calculated automatically
        fmap_type:{'scatter', 'grid'}, default: 'gird', if 'scatter', will return a scatter mol map without an assignment to a grid
        split_channels: bool, if True, outputs will split into various channels using the types of feature
        metric: {'cosine', 'correlation'}, default: 'cosine', measurement of feature distance
        var_thr: float, defalt is 1e-4, meaning that feature will be included only if the conresponding variance larger than this value. Since some of the feature has pretty low variances, we can remove them by increasing this threshold
        """
        
        super().__init__()
        assert fmap_type in ['scatter', 'grid'], 'no such feature map type supported!'
       
        self.ftype = "omics"
        self.metric = metric
        self.method = None
        self.isfit = False

        #default we will load the  precomputed matrix

        scale_info = pd.read_pickle("%s\\omics_scale.cfg"%path)
        with h5py.File('%s\\omics_cosine.h5'%path,'r') as f:
            _dist_matrix = f['omics_cosine'][:]
        dist_matrix = pd.DataFrame(_dist_matrix, index=scale_info.index.values, columns=scale_info.index.values)
        flist = list(dist_matrix.columns)
        self.dist_matrix = dist_matrix
        self.flist = list(dist_matrix.columns)
        self.scale_info = scale_info
        self.extract = Extraction(path = path, omics_colormaps = omics_colormaps) 
        self.fmap_type = fmap_type
        self.flist = flist
        
        if fmap_type == 'grid':
            S = Scatter2Grid()
        else:
            if fmap_shape == None:
                N = len(self.flist)
                l = np.int(np.sqrt(N))*2
                fmap_shape = (l, l)                
            S = Scatter2Array(fmap_shape)
        
        self._S = S
        self.split_channels = split_channels        

        
        
    def _fit_embedding(self, 
                        method = 'tsne',  
                        n_components = 2,
                        random_state = 1,  
                        verbose = 2,
                        n_neighbors = 30,
                        min_dist = 0.1,
                        **kwargs):
        
        """
        parameters
        -----------------
        method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding algorithm
        """
        dist_matrix = self.dist_matrix
        if 'metric' in kwargs.keys():
            metric = kwargs.get('metric')
            kwargs.pop('metric')
            
        else:
            metric = 'precomputed'

        if method == 'tsne':
            embedded = TSNE(n_components=n_components, 
                            random_state=random_state,
                            metric = metric,
                            verbose = verbose,
                            **kwargs)
        elif method == 'umap':
            embedded = UMAP(n_components = n_components, 
                            n_neighbors = n_neighbors,
                            min_dist = min_dist,
                            verbose = verbose,
                            random_state=random_state, 
                            metric = metric, **kwargs)
            
        elif method =='mds':
            if 'metric' in kwargs.keys():
                kwargs.pop('metric')
            if 'dissimilarity' in kwargs.keys():
                dissimilarity = kwargs.get('dissimilarity')
                kwargs.pop('dissimilarity')
            else:
                dissimilarity = 'precomputed'
                
            embedded = MDS(metric = True, 
                           n_components= n_components,
                           verbose = verbose,
                           dissimilarity = dissimilarity, 
                           random_state = random_state, **kwargs)
        
        embedded = embedded.fit(dist_matrix)    
        df = pd.DataFrame(embedded.embedding_, index = self.flist,columns=['x', 'y'])
        typemap = self.extract.bitsinfo.set_index('IDs')
        df = pd.concat((df,typemap),axis=1)
        df['Channels'] = df['Subtypes']
        self.df_embedding = df
        self.embedded = embedded
        

    def fit(self, 
            method = 'umap', min_dist = 0.1, n_neighbors = 30,
            verbose = 2, random_state = 1, **kwargs): 
        """
        parameters
        -----------------
        method: {'tsne', 'umap', 'mds'}, algorithm to embedd high-D to 2D
        kwargs: the extra parameters for the conresponding method
        """
        if 'n_components' in kwargs.keys():
            kwargs.pop('n_components')
            
        ## embedding  into a 2d 
        assert method in ['tsne', 'umap', 'mds'], 'no support such method!'
        
        self.method = method
        
        ## 2d embedding first
        self._fit_embedding(method = method,
                            n_neighbors = n_neighbors,
                            random_state = random_state,
                            min_dist = min_dist, 
                            verbose = verbose,
                            n_components = 2, **kwargs)

        
        if self.fmap_type == 'scatter':
            ## naive scatter algorithm
            print_info('Applying naive scatter feature map...')
            self._S.fit(self.df_embedding, self.split_channels, channel_col = 'Channels')
            print_info('Finished')
            
        else:
            ## linear assignment algorithm 
            print_info('Applying grid feature map(assignment), this may take several minutes(1~30 min)')
            self._S.fit(self.df_embedding, self.split_channels, channel_col = 'Channels')
            print_info('Finished')
        
        ## fit flag
        self.isfit = True
        self.fmap_shape = self._S.fmap_shape
    

    
    def transform(self, 
                  sigle_npy, 
                ):
    
    
        """
        parameters
        --------------------
        smiles:smiles string of compound
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        """
        
        if not self.isfit:
            print_error('please fit first!')
            return

        arr = sigle_npy
        df = pd.DataFrame(arr).T
        df.columns = self.extract.bitsinfo.IDs
        vector_1d = df.values[0] #shape = (N, )
        fmap = self._S.transform(vector_1d)    
        return np.nan_to_num(fmap)
        

        
    def batch_transform(self, 
                        npy, 
                        n_jobs=4):
    
        """
        parameters
        --------------------
        smiles_list: list of smiles strings
        scale: bool, if True, we will apply MinMax scaling by the precomputed values
        scale_method: {'minmax', 'standard'}
        n_jobs: number of parallel
        """
        
        P = Parallel(n_jobs=n_jobs)
        res = P(delayed(self.transform)(sigle_npy) for sigle_npy in tqdm(npy, ascii=True)) 
        X = np.stack(res) 
        X = X.transpose(0,3,1,2)

        return X

    
    def rearrangement(self, orignal_X, target_mp):

        """
        Re-Arragement feature maps X from orignal_mp's to target_mp's style, in case that feature already extracted but the position need to be refit and rearrangement.

        parameters
        -------------------
        orignal_X: the feature values transformed from orignal_mp(object self)
        target_mp: the target feature map object

        return
        -------------
        target_X, shape is (N, W, H, C)
        """
        assert self.flist == target_mp.flist, print_error('Input features list is different, can not re-arrangement, check your flist by mp.flist method' )
        assert len(orignal_X.shape) == 4, print_error('Input X has error shape, please reshape to (samples, w, h, channels)')
        
        idx = self._S.df.sort_values('indices').idx.tolist()
        idx = np.argsort(idx)

        N = len(orignal_X) #number of sample
        M = len(self.flist) # number of features
        res = []
        for i in tqdm(range(N), ascii=True):
            x = orignal_X[i].sum(axis=-1)
            vector_1d_ordered = x.reshape(-1,)
            vector_1d_ordered = vector_1d_ordered[:M]
            vector_1d = vector_1d_ordered[idx]
            fmap = target_mp._S.transform(vector_1d)
            res.append(fmap)
        return np.stack(res)

    
    
    def plot_scatter(self, htmlpath='./', htmlname=None, radius = 3):
        """radius: the size of the scatter, must be int"""
        df_scatter, H_scatter = vismap.plot_scatter(self,  
                                htmlpath=htmlpath, 
                                htmlname=htmlname,
                                radius = radius)
        
        self.df_scatter = df_scatter
        return H_scatter   
        
        
    def plot_grid(self, htmlpath='./', htmlname=None):
        
        if self.fmap_type != 'grid':
            return
        
        df_grid, H_grid = vismap.plot_grid(self,  
                                htmlpath=htmlpath, 
                                htmlname=htmlname)
        
        self.df_grid = df_grid
        return H_grid       
        
        
    def load(self, filename):
        return self._load(filename)
    
    
    def save(self, filename):
        return self._save(filename)
    