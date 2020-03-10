#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Tue Mar 10 15:04:11 2020
@author: Qian Zhu
@modify: Minsheng Hao
"""

import scipy
import scipy.stats
import sys
import re
import os
import numpy as np
import math
from operator import itemgetter
from scipy.spatial.distance import squareform, pdist
from scipy.stats import percentileofscore
#import smfishHmrf.reader as reader
import pandas as pd
import parmap
from tqdm import tqdm

def get_distance_per_FD_2(mr_dissimilarity_FD, num_cell, clust, outcome=[1,2]):
    c1 = np.where(clust==1)[0]
    c2 = np.where(clust==2)[0]
    within_dist = mr_dissimilarity_FD[np.ix_(c1, c1)]
    across_dist = mr_dissimilarity_FD[np.ix_(c1, c2)]
    mm_vec = (np.sum(within_dist, axis=1) - within_dist.diagonal()) / float(within_dist.shape[0] - 1)
    mn_vec = np.mean(across_dist, axis=1)
    sil_vec = (mn_vec - mm_vec)/np.max(np.concatenate(([mn_vec], [mm_vec])), axis=0)
    avg_clust1_sil = np.mean(sil_vec)
    return avg_clust1_sil

def rank_transform_matrix(mat, rbp_p = 0.99, reverse=True):
    dim1 = mat.shape[0]
    dim2 = mat.shape[1]
    rank = np.empty([dim1, dim2])

    print("Start ranking...")
    for c1 in range(dim1):
        rd = scipy.stats.rankdata(mat[c1,:])
        if reverse==True:
            rd = dim2 - rd + 1
        rank[c1, :] = rd
        if c1%1000==0:
            print("Done %d" % c1)

    print("Finished ranking...")
    mutual_rank_rbp = np.empty([dim1, dim2])
    mutual_rank = np.empty([dim1, dim2])


    print("Calculate mutual rank...")
    ma = np.sqrt(np.multiply(rank, rank.T))
    print("Calculate dissimilarity...")
    dissimilarity = np.subtract(1, np.power(rbp_p, np.subtract(ma, 1)))
    print("Finished dissimilarity...")
    return dissimilarity

def process(expr,ncell,genes,examine_top,dissim):
    sil_sub=[]
    ex = int((1.0-examine_top)*100.0)
    for ig,g in enumerate(genes):
        cutoff = np.percentile(expr[ig,:], ex)
        clust = np.zeros((ncell), dtype="int32")
        gt_eq = np.where(expr[ig,:]>=cutoff)[0]
        lt = np.where(expr[ig,:]<cutoff)[0]
        if gt_eq.shape[0]>int(ncell*examine_top):
            num_filter = gt_eq.shape[0] - int(ncell*examine_top)
            ss = np.random.choice(gt_eq, size=num_filter, replace=False)
            clust[gt_eq] = 1
            clust[lt] = 2
            clust[ss] = 2
        elif gt_eq.shape[0]<int(ncell*examine_top):
            num_filter = int(ncell*examine_top) - gt_eq.shape[0]
            ss = np.random.choice(lt, size=num_filter, replace=False)
            clust[gt_eq] = 1
            clust[lt] = 2
            clust[ss] = 1
        else:
            clust[gt_eq] = 1
            clust[lt] = 2
        avg_clust1_sil = get_distance_per_FD_2(dissim, ncell, clust, outcome=[1,2])
#         tmpsil = []
#         for k in range(300):
#             np.random.shuffle(clust)
#             tmpsil.append(get_distance_per_FD_2(dissim, ncell, clust, outcome=[1,2]))
#         p = est(avg_clust1_sil,tmpsil)
        sil_sub.append((g, -1, avg_clust1_sil))
    return sil_sub

def calc_silhouette_per_gene(genes=None, expr=None, dissim=None, examine_top=0.1, seed=-1,num_cores=8,bar=True):
    if genes is None or expr is None or dissim is None:
        sys.stderr.write("Need genes, expr, dissim\n")
        return ;
    if seed!=-1 and seed>=0:
        np.random.seed(seed)
    sys.stdout.write("Started 2 " + "\n")
    sil = []
    ncell = expr.shape[1]
    ex = int((1.0-examine_top)*100.0)
    subexpr = np.array_split(expr,num_cores,axis=0)
    subgenes = np.array_split(np.array(genes),num_cores)
    subgenes = [i.tolist() for i in subgenes]
    print('number of cores : '+str(num_cores))
    dis_list=[]
    for i in range(num_cores):
        dis_list.append(dissim)
    assert len(subexpr)==len(subgenes)
    tuples = [(expr, ncell, genes, exa, dis) for expr, ncell, genes, exa, dis in zip(subexpr, repeat(ncell, num_cores),
                                    subgenes,
                                    repeat(examine_top, num_cores), 
                                    dis_list)] 
    results = parmap.starmap(process, tuples,
                             pm_processes=num_cores, pm_pbar=bar)
    for i in np.arange(len(results)):
        sil += results[i]
    
    res = []
    for ig,g in enumerate(genes):
        this_avg = sil[ig][1]
        this_sil = sil[ig][2]
        res.append((g, this_sil))
    res.sort(key=itemgetter(1), reverse=True)
    return res


def python_spatial_genes(spatial_locations, expression_matrix,
                         metric = "euclidean",
                         rbp_p = 0.95, examine_top = 0.3, seed=-1,cpus=8,bar=True):

    Xcen =  spatial_locations
    mat = expression_matrix
    genes = []

    for g in range(mat.index.shape[0]):
        genes.append(str(mat.index[g]))
    expr = np.copy(mat.values)

    ncell = Xcen.shape[0] 
    sys.stdout.write("Calculate all pairwise Euclidean distance between cells using their physical coordinates\n")
    euc = squareform(pdist(Xcen, metric=metric))
    sys.stdout.write("Rank transform euclidean distance, and then apply exponential transform\n")
    dissim = rank_transform_matrix(euc, reverse=False, rbp_p=rbp_p)
    sys.stdout.write("Compute silhouette metric per gene\n")
    res = calc_silhouette_per_gene(genes=genes, expr=expr, dissim=dissim, examine_top=examine_top,seed=seed,num_cores=cpus,bar=bar)

    return res

