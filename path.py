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
N_sample = 400      # Number of sample points
N_knn = 20          # Number of neighbours of each point
max_edge_len = 20.0 # Maximum edge length
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
  if dist <=pt_size:
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

      if not collision(ix, iy, nx, ny, pt_size, obstacle_kdtree):
        edge_id.append(indexes[z])

      if (len(edge_id) >= N_knn):
        break

    roadmap.append(edge_id)
  return roadmap

def PRM_planning(x_start, y_start, x_goal, y_goal, env, pt_size):
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
  sample_x, sample_y = sample_pts(x_start, y_start,x_goal, y_goal, env)

  roadmap = road_map(sample_x, sample_y, pt_size, obstacle_kdtree)
  #print(roadmap)
  px, py = dij_planning(x_start, y_start, x_goal, y_goal, roadmap, sample_x, sample_y)

  return roadmap, sample_x ,sample_y, px ,py

def plt_roadmap(roadmap, sample_x, sample_y):

    for i, _ in enumerate(roadmap):
        for ii in range(len(roadmap[i])):
            ind = roadmap[i][ii]

            pl.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")

#################### Dijkstra's Algorithm ####################
# Node class for Dijkstra's Algorithm
class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

def dij_planning(sx, sy, gx, gy, road_map, sample_x, sample_y):

    nstart = Node(sx, sy, 0.0, -1)
    ngoal = Node(gx, gy, 0.0, -1)

    openset, closedset = dict(), dict()
    openset[len(road_map) - 2] = nstart

    while True:
        if not openset:
            print("Cannot find path...")
            break

        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]

        # show graph
        if len(closedset.keys()) % 2 == 0:
            pl.plot(current.x, current.y, "xg")
            pl.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("Goal is found!")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.sqrt(dx**2 + dy**2)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closedset:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:
                openset[n_id] = node

    # generate final course
    rx, ry = [ngoal.x], [ngoal.y]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        ry.append(n.y)
        pind = n.pind

    return rx, ry

##### Calling PRM Path Planning #####
map,x,y,px,py = PRM_planning(x_start, y_start, x_goal, y_goal, env, pt_size)
#plt_roadmap(map,x,y)
pl.scatter(x,y)
pl.plot(px,py, "-g", linewidth = 2.0)
pl.pause(100)