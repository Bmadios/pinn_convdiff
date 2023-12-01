# -*- coding: utf-8 -*-
"""
Code file utils.py 

@author: Blaise Madiega
email : blaisemadiega@gmail.com
"""

import math
import torch
import random
import numpy as np
import pandas as pd

##CUDA Support
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

t_min = 0.0
t_max = 0.61
deltaT = 0.01

def exp_decay_schedule(epoch, initial_lr, decay_rate, plateau_epoch):
    """
    Calculates the exponentially decaying learning rate for each epoch.

    Args:
        epoch (int): The current training epoch.
        initial_lr (float): The initial learning rate.
        decay_rate (float): The decay rate for the learning rate.
        plateau_epoch (int): The epoch after which learning rate decay begins.

    Returns:
        float: The learning rate for the current epoch.
    """
    lr = initial_lr * math.exp(-decay_rate * (epoch / plateau_epoch))
    return max(lr, 1e-5)

def init_weights(m):
    """
    Initializes the weights and biases of the provided model.

    Args:
        m (torch.nn.Module): The model whose weights and biases should be initialized.

    Returns:
        None
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)

def colloc_pde_interior(N_x, N_y, x_min, x_max, y_min, y_max):
    """
    Generates collocation points (x, y) for each time step and returns those that are outside a specified boundary.

    Args:
        N_x (int): Number of points in x-direction.
        N_y (int): Number of points in y-direction.
        x_min (float): Minimum x-coordinate.
        x_max (float): Maximum x-coordinate.
        y_min (float): Minimum y-coordinate.
        y_max (float): Maximum y-coordinate.

    Returns:
        tensor: Generated interior collocation points that are outside the specified boundary.
    """
    x_values = np.linspace(x_min, x_max, N_x)
    y_values = np.linspace(y_min, y_max, N_y)
    t_values = np.arange(t_min, t_max, deltaT)
    
    txy = []
    for t in t_values:
        for x in x_values:
            for y in y_values:
                if not (y == y_min or y == y_max or x == x_min or x == x_max):
                    txy.append([t, x, y])
                    
    txy = torch.from_numpy(np.array(txy)).float()
    return txy

def colloc_pde_with_mesh(N_x, N_y, x_min, x_max, y_min, y_max):
  """
    Generates random interior collocation points using MESHING of domain strategy.

    Args:
        N_x (int): Number of points in x-direction.
        N_y (int): Number of points in y-direction.
        x_min (float): Minimum x-coordinate.
        x_max (float): Maximum x-coordinate.
        y_min (float): Minimum y-coordinate.
        y_max (float): Maximum y-coordinate.

    Returns:
        tensor: Randomly generated interior collocation points that are outside the specified boundary.
    """
  txy = []
  for i in range(N_x*N_y):
    t_values = np.arange(t_min, t_max, deltaT)
    t = np.random.choice(t_values)
    x_1 = random.uniform(x_min, x_max/2)
    y_1 = random.uniform(y_min, y_max/2) 
    x_2 = random.uniform(x_max/2, x_max)
    y_2 = random.uniform(y_min, y_max/2) 
    x_3 = random.uniform(x_max/2, x_max)
    y_3 = random.uniform(y_max/2, y_max) 
    x_4 = random.uniform(x_min, x_max/2)
    y_4 = random.uniform(y_max/2, y_max) 
    txy.append([t, x_1, y_1])
    txy.append([t, x_2, y_2])
    txy.append([t, x_3, y_3])
    txy.append([t, x_4, y_4])
  
  indices = []
  x_inlet = x_min
  x_outlet = x_max
  y_wall_bottom = y_min
  y_wall_top = y_max
  #compteur = 0
  for i in range(len(txy)):
    if txy[i][2] == y_wall_bottom or txy[i][2] == y_wall_top or txy[i][1] == x_inlet or txy[i][1] == x_outlet:

      indices.append(i)

  txy = np.delete(txy, indices, axis = 0)
  txy = torch.from_numpy(txy).float().to(device)
  return txy

def colloc_Xinlet(Nin, x_min, y_min, y_max):
  """
  Generates the inlet collocation points.
  
  Args:
    Nin (int): The square root of the total number of collocation points.
    x_min (float): The minimum x coordinate of the collocation points.
    y_min (float): The minimum y coordinate of the collocation points.
    y_max (float): The maximum y coordinate of the collocation points.
  
  Returns:
    torch.tensor: The collocation points as a torch tensor.
  """
  txy = []
  #x = np.linspace(x_min, x_max, N_x)
  y_values = np.linspace(y_min, y_max, Nin)
  t_values = np.arange(t_min, t_max, deltaT)
  for t in t_values:
    for y in y_values:
        txy.append([t, x_min, y])
                    
  txy = torch.from_numpy(np.array(txy)).float()
  return txy

def colloc_Xoutlet(Nout, x_max, y_min, y_max):
  """
  Generates the outlet collocation points.
  
  Args:
    Nout (int): The square root of the total number of collocation points.
    x_min (float): The minimum x coordinate of the collocation points.
    y_min (float): The minimum y coordinate of the collocation points.
    y_max (float): The maximum y coordinate of the collocation points.
  
  Returns:
    torch.tensor: The collocation points as a torch tensor.
  """
  txy = []
  #x = np.linspace(x_min, x_max, N_x)
  y_values = np.linspace(y_min, y_max, Nout)
  t_values = np.arange(t_min, t_max, deltaT)
  for t in t_values:
    for y in y_values:
        txy.append([t, x_max, y])
                    
  txy = torch.from_numpy(np.array(txy)).float()
  return txy

def colloc_Wall_bottom(Nwallbot, x_min, x_max, y_min):
  """
  Generates the bottom wall collocation points.
  
  Args:
    Nwallbot (int): The square root of the total number of collocation points.
    x_min (float): The minimum x coordinate of the collocation points.
    x_max (float): The maximum x coordinate of the collocation points.
    y_min (float): The y coordinate of the bottom wall.
  
  Returns:
    torch.tensor: The collocation points as a torch tensor.
  """
  txy = []
  x_values = np.linspace(x_min, x_max, Nwallbot)
  #y_values = np.linspace(y_min, y_max, Nin)
  t_values = np.arange(t_min, t_max, deltaT)
  for t in t_values:
    for x in x_values:
        txy.append([t, x, y_min])
                    
  txy = torch.from_numpy(np.array(txy)).float()
  return txy

def colloc_Wall_top(Nwalltop, x_min, x_max, y_max):
  """
  Generates the top wall collocation points.
  
  Args:
    Nwalltop (int): The square root of the total number of collocation points.
    x_min (float): The minimum x coordinate of the collocation points.
    x_max (float): The maximum x coordinate of the collocation points.
    y_min (float): The y coordinate of the bottom wall.
  
  Returns:
    torch.tensor: The collocation points as a torch tensor.
  """
  txy = []
  x_values = np.linspace(x_min, x_max, Nwalltop)
  #y_values = np.linspace(y_min, y_max, Nin)
  t_values = np.arange(t_min, t_max, deltaT)
  for t in t_values:
    for x in x_values:
        txy.append([t, x, y_max])
                    
  txy = torch.from_numpy(np.array(txy)).float()
  return txy


def load_data(file_path, N_mes, device):
    """Loads and preprocesses data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        N_mes (int): The number of random samples to take from the data.
        device (str): The device to send the tensors to.

    Returns:
        Tuple: Tuple containing:
            random_XY_data (torch.Tensor): Random samples of XY data.
            XY_data (torch.Tensor): All XY data.
            random_UVP_data (torch.Tensor): Random samples of UVP data.
            UVP_data (torch.Tensor): All UVP data.
            BC_XY_data (torch.Tensor): Filtered XY data.
            BC_UVP_data (torch.Tensor): Filtered UVP data.
    """
    data = pd.read_csv(file_path)

    T_dat = data["temps"].values
    X_dat = data["x"].values
    Y_dat = data["y"].values
    u_dat = data["u"].values

    txy = np.array([[t, x, y] for t, x, y in zip(T_dat, X_dat, Y_dat)])
    u_data = np.array([[u] for u in zip(u_dat)])

    random_indices = random.sample(range(len(txy)), N_mes)

    random_TXY_data = torch.from_numpy(txy[random_indices]).float().to(device)
    random_U_data = torch.from_numpy(u_data[random_indices]).float().to(device)

    TXY_data = torch.from_numpy(txy).float().to(device)
    U_data = torch.from_numpy(u_data).float().to(device)

    BC_indices = [i for i, (t, x, y) in enumerate(txy) if x in {0, 2} or y in {0, 2}]

    BC_TXY_data = torch.from_numpy(txy[BC_indices]).float().to(device)
    BC_U_data = torch.from_numpy(u_data[BC_indices]).float().to(device)

    #print(TXY_data)
    #print(U_data)

    return random_TXY_data, TXY_data, random_U_data, U_data, BC_TXY_data, BC_U_data
