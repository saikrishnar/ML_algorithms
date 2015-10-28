#!/usr/bin/python
## This implementention has been emulated from the wordpress blog datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python

##### Author : R Sai Krishna , 
#####          Speech and Vision Lab, 
#####          International Institute of Information Technology- Hyderabad. 
##### Date   : 27 October 2015


import random
import numpy as np


############################ Initialization ##################################

## Random Initialization
def init_board(N):
    X = np.array([(random.uniform(-1,1), random.uniform(-1,1)) for i in range(N)])
    return X

## Initialization with Gaussian distributed clusters with random variances
def init_board_gauss(N,k):
    n = float(N)/k
    X = []
    for i in range(K):
       c = (random.uniform(-1,1), random.uniform(-1,1))
       s = ramdon.uniform(0.05,0.5)
       x = []
       while len(x) < n:
          a,b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
          # Continue drawing points from the distribution in the range [-1,1]
          if abs(a) < 1 and abs(b) < 1 :
             x.append(a,b)
       X.extend(x)
    X = np.array(X)[:N]
    return X
  
################################################################################


## Checking Condition
def has_converged(mu,oldmu):
    return(set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

## Expectation Step
def cluster_points(X, mu):
    clusters = {}
    for x in X:
#         bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) for i in enumerate(mu) ], key = lambda t:t[1])[0]
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]   
        try:
           clusters[bestmukey].append(x)
        except KeyError:
           clusters[bestmukey] = [x]
    return clusters

## Maximization Step
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu


## Main Algorithm
def my_kmeans(K, X):
    # Initialize to K random centroids
    oldmu = random.sample(X,K)
    mu = random.sample(X,K)
    while not has_converged(mu,oldmu):
        oldmu = mu
        # Assign all the points in X to clusters
        clusters = cluster_points(X,mu)
        # Re-evaluate clusters
        mu = reevaluate_centers(oldmu,clusters)        
    return(mu,clusters)
         

def main():
    N = 100
    X = init_board(N)
    K = 7
    [mu,clusters] = my_kmeans(K,X)
    print mu
    print clusters

if __name__ == '__main__':
    main()
