#!/usr/bin/env python
import random
import math
import numpy as np
import scipy.spatial #To find neasrest neighbours
from scipy.spatial import KDTree
import pylab as pl
import sys
sys.path.append('osr_examples/scripts/')
import environment_2d

########## Parameter ##########
N_sample = 200        # Number of sample points
N_knn = 15            # Number of neighbours of each point
max_edge_len = 20.0   # Maximum edge length
pt_size = 0.01

pl.ion()
np.random.seed(4)
env = environment_2d.Environment(10, 6, 5)
pl.clf()
env.plot()
q = env.random_query()
if q is not None:
  x_start, y_start, x_goal, y_goal = q
  env.plot_query(x_start, y_start, x_goal, y_goal)

#################### PRM Path Planning ####################
def sample_pts(x_start, y_start,x_goal, y_goal, env):

  sample_x, sample_y = [], []

  while len(sample_x) <= N_sample:
    tx =(random.random() * (env.size_x-0)) + 0
    ty =(random.random() * (env.size_y-0)) + 0

    if env.check_collision(tx,ty)==False:
      sample_x.append(tx)
      sample_y.append(ty)
  
  sample_x.append(x_start)
  sample_y.append(y_start)
  sample_x.append(x_goal)
  sample_y.append(y_goal)
  
  return sample_x, sample_y

def collision(x0,y0,x1,y1,pt_size, obstacle_kdtree):
  dx = x1 - x0
  dy = y1 - y0
  yaw = math.atan2(dy,dx)
  norm = math.sqrt(dx**2 + dy**2)
  
  if norm >= max_edge_len:
    return True

  D = pt_size
  n_step = round(norm / D)

  for i in range (int(n_step)):
    dist, _ = obstacle_kdtree.query([x0,y0])
    if dist <= pt_size:
      return True
    x0 += D * math.cos(yaw)
    y0 += D * math.sin(yaw)
  
  #Goal point check
  dist, _= obstacle_kdtree.query([x1,y1])
  if dist<=pt_size:
    return True

  return False

def road_map(sample_x, sample_y, pt_size, obstacle_kdtree):
  roadmap = []
  n_sample = len(sample_x)
  sample_kdtree = KDTree(np.vstack((sample_x, sample_y)).T)

  for (i,ix,iy) in zip(range(n_sample),sample_x, sample_y):
    dists, indexes = sample_kdtree.query([ix,iy], k = n_sample)
    edge_id = []

    for z in range(1,len(indexes)):
      nx = sample_x[indexes[z]]
      ny = sample_y[indexes[z]]

      if not collision(ix, iy, nx, ny, pt_size, obstacle_kdtree): #Check if collison occurs
        edge_id.append(indexes[z])

      if (len(edge_id) >= N_knn):
        break

    roadmap.append(edge_id)

  return roadmap

def PRM_planning(x_start, y_start, x_goal, y_goal,env, pt_size):
  ox, oy = [], [] # To get all the points which form the obstables (Triangles)
  for i in range(len(env.obs)):
    x,y = [],[]
    x.append(env.obs[i].x0)
    x.append(env.obs[i].x1)
    x.append(env.obs[i].x2)
    x.append(x[0])
    y.append(env.obs[i].y0)
    y.append(env.obs[i].y1)
    y.append(env.obs[i].y2)
    y.append(y[0])
    m,c = [0,0,0], [0,0,0]
    for l in range(3):
      m[l] = (y[l+1]-y[l]) / (x[l+1]-x[l])
      c[l] = y[l] - m[l] * x[l]
      X = min(x[l], x[l+1])
      Y = min(y[l], x[l+1])
      if (abs(x[l+1] - x[l]) <=1):
        while (Y <= max(y[l], y[l+1])):
          X = (Y - c[l]) / m[l]
          ox.append(X)
          oy.append(Y)
          Y += 0.001
      else:
          while (X <= max(x[l], x[l+1])):
            Y = m[l] * X + c[l]
            ox.append(X)
            oy.append(Y)
            X += 0.001
  #ox.append(0.0)
  #ox.append(env.size_x)
  #oy.append(0.0)
  #oy.append(env.size_y)

  obstacle_kdtree = KDTree(np.vstack((ox,oy)).T)
  sample_x, sample_y = sample_pts(x_start, y_start, x_goal, y_goal, env)

  roadmap = road_map(sample_x, sample_y, pt_size, obstacle_kdtree)
  
  return roadmap, sample_x ,sample_y

def plt_roadmap(roadmap, sample_x, sample_y):

    for i, _ in enumerate(roadmap):
        for ii in range(len(roadmap[i])):
            ind = roadmap[i][ii]

            pl.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")

########## Calling PRM_planning and Plotting ##########
map,x,y = PRM_planning(x_start, y_start, x_goal, y_goal, env, pt_size)
plt_roadmap(map,x,y)
pl.scatter(x,y)
pl.pause(100)




########## Post-processing ##########
""" 
Algorithm: PATH_SHORTCUTTING
Input: A collision-free path q
Effect: Decrease the path length of q
"""
#for rep = 1 in range(100):