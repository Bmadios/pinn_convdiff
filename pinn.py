# -*- coding: utf-8 -*-
"""
PINN Python Code pinn.py

@author: Blaise Madiega
email : blaisemadiega@gmail.com
"""

import torch
from torch.nn import Module, Linear, Tanh
from collections import OrderedDict
from utilis import load_data

##CUDA Support
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
  

class PINN_NavierStokes(Module):
    def __init__(self, layers):
        """
        Initialize the PINN model.
        
        Args:
        layers (list): A list specifying the number of units in each layer.
        """
        super(PINN_NavierStokes, self).__init__()

        self.depth = len(layers) - 1 
        self.activation = Tanh() 

        layer_list = []
        for i in range(self.depth-1):
            layer_list.append(('layer_%d' % i, Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation))
        layer_list.append(('layer_%d' % (self.depth-1), Linear(layers[-2], layers[-1])))

        self.layers = torch.nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
        x (Tensor): The input tensor.
        
        Returns:
        Tensor: The output tensor.
        """
        return self.layers(x)

def pde_loss(u_net, TXY, nu, c):
  #x = torch.tensor(XY[:,0], requires_grad = True).float().cpu()
  #y = torch.tensor(XY[:, 1], requires_grad = True).float().cpu()
  t = TXY[:, 0].clone().detach().requires_grad_(True).float().cpu()
  x = TXY[:, 1].clone().detach().requires_grad_(True).float().cpu()
  y = TXY[:, 2].clone().detach().requires_grad_(True).float().cpu()
  txy_pde = torch.hstack((torch.reshape(t, (-1, 1)), torch.reshape(x, (-1, 1)), torch.reshape(y, (-1, 1)))).float().to(device)

  u = u_net(txy_pde)[:, 0].to(device) # Velocity u

  t.to(device)
  x.to(device)
  y.to(device)

  u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True , create_graph=True)[0].to(device)
  u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True , create_graph=True)[0].to(device)
  u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True , create_graph=True)[0].to(device)
  u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True , create_graph=True)[0].to(device)
  u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True , create_graph=True)[0].to(device)



  # Loss computing
  f1 = u_t + c*(u_x + u_y) - nu*(u_xx + u_yy) # CONVECTION-DIFFUSION 2D PDE RESIDUALS

  loss = torch.mean(f1**2)

  return loss

def boundaries_loss(u_net, TXY_INLET, TXY_OUTLET, TXY_WBOTTOM, TXY_WTOP, U_BC):
  #x = torch.tensor(XY[:,0], requires_grad = True).float().cpu()
  #y = torch.tensor(XY[:, 1], requires_grad = True).float().cpu()
  t_in = TXY_INLET[:, 0].clone().detach().requires_grad_(True).float().cpu()
  x_in = TXY_INLET[:, 1].clone().detach().requires_grad_(True).float().cpu()
  y_in = TXY_INLET[:, 2].clone().detach().requires_grad_(True).float().cpu()
  txy_in = torch.hstack((torch.reshape(t_in, (-1, 1)), torch.reshape(x_in, (-1, 1)), torch.reshape(y_in, (-1, 1)))).float().to(device)

  t_out = TXY_OUTLET[:, 0].clone().detach().requires_grad_(True).float().cpu()
  x_out = TXY_OUTLET[:, 1].clone().detach().requires_grad_(True).float().cpu()
  y_out = TXY_OUTLET[:, 2].clone().detach().requires_grad_(True).float().cpu()
  txy_out = torch.hstack((torch.reshape(t_out, (-1, 1)), torch.reshape(x_out, (-1, 1)), torch.reshape(y_out, (-1, 1)))).float().to(device)

  t_bot = TXY_WBOTTOM[:, 0].clone().detach().requires_grad_(True).float().cpu()
  x_bot = TXY_WBOTTOM[:, 1].clone().detach().requires_grad_(True).float().cpu()
  y_bot = TXY_WBOTTOM[:, 2].clone().detach().requires_grad_(True).float().cpu()
  txy_bot = torch.hstack((torch.reshape(t_bot, (-1, 1)), torch.reshape(x_bot, (-1, 1)), torch.reshape(y_bot, (-1, 1)))).float().to(device)

  t_top= TXY_WTOP[:, 0].clone().detach().requires_grad_(True).float().cpu()
  x_top = TXY_WTOP[:, 1].clone().detach().requires_grad_(True).float().cpu()
  y_top = TXY_WTOP[:, 2].clone().detach().requires_grad_(True).float().cpu()
  txy_top = torch.hstack((torch.reshape(t_top, (-1, 1)), torch.reshape(x_top, (-1, 1)), torch.reshape(y_top, (-1, 1)))).float().to(device)


  u_in = u_net(txy_in)[:, 0].to(device) # Velocity u_in

  u_out = u_net(txy_out)[:, 0].to(device) # Velocity u_out

  u_bot = u_net(txy_bot)[:, 0].to(device) # Velocity u_wall bottom

  u_top = u_net(txy_top)[:, 0].to(device) # Velocity u_wall top


  t_in.to(device)
  x_in.to(device)
  y_in.to(device)

  t_out.to(device)
  x_out.to(device)
  y_out.to(device)

  t_bot.to(device)
  x_bot.to(device)
  y_bot.to(device)

  t_top.to(device)
  x_top.to(device)
  y_top.to(device)
    
    
  ##### GRADIENTS NEUMANN BCs ###########
  u_x_out = torch.autograd.grad(u_out, x_out, grad_outputs=torch.ones_like(u_out), retain_graph=True , create_graph=True)[0].to(device)
  u_y_out = torch.autograd.grad(u_out, y_out, grad_outputs=torch.ones_like(u_out), retain_graph=True , create_graph=True)[0].to(device)
  u_y_top = torch.autograd.grad(u_top, y_top, grad_outputs=torch.ones_like(u_top), retain_graph=True , create_graph=True)[0].to(device)
  u_x_top = torch.autograd.grad(u_top, x_top, grad_outputs=torch.ones_like(u_top), retain_graph=True , create_graph=True)[0].to(device)

  # Loss computing
  fbc_inlet_u = u_in - U_BC # inlet u
  fbc_outlet_u = u_x_out
  #fbc_outlet_u = u_y_out  # x = 2# x = 2
  fbc_bottom_u = u_bot - U_BC  # ux bottom = 0
  fbc_top_u = u_y_top # y = 2
  #fbc_top_u = u_x_top
  

  #loss_BC_inlet = torch.mean(fbc_inlet_u**2) + torch.mean(fbc_inlet_v**2) + torch.mean(fbc_inlet_p1**2) + torch.mean(fbc_inlet_p2**2)
  loss_BC_inlet = torch.mean(fbc_inlet_u**2) # + torch.mean(fbc_inlet_p**2)
  #loss_BC_outlet = torch.mean(fbc_outlet_u1**2) + torch.mean(fbc_outlet_u2**2) + torch.mean(fbc_outlet_v1**2) + torch.mean(fbc_outlet_v2**2) +    torch.mean(fbc_outlet_p**2)
  loss_BC_outlet = torch.mean(fbc_outlet_u**2)  # + torch.mean(fbc_outlet_p**2)
  #loss_BC_bottom = torch.mean(fbc_bottom_u**2) + torch.mean(fbc_bottom_v**2) + torch.mean(fbc_bottom_p1**2) + torch.mean(fbc_bottom_p2**2)
  loss_BC_bottom = torch.mean(fbc_bottom_u**2) # + torch.mean(fbc_bottom_p**2)
  #loss_BC_top = torch.mean(fbc_top_u**2) + torch.mean(fbc_top_v**2) + torch.mean(fbc_top_p1**2) + torch.mean(fbc_top_p2**2)
  loss_BC_top = torch.mean(fbc_top_u**2) # + torch.mean(fbc_top_p**2)
  #loss_BC_c = torch.mean(fbc_c_u**2) + torch.mean(fbc_c_v**2) + torch.mean(fbc_c_p1**2) + torch.mean(fbc_c_p2**2)

  total_loss = loss_BC_inlet + loss_BC_outlet + loss_BC_bottom + loss_BC_top
  return total_loss


def initial_condition_loss(u_net, initial_TXY, initial_U):
    """
    Compute the loss associated with the initial condition.

    Args:
    - u_net: The neural network model.
    - initial_TXY: Tensor containing the initial time and spatial locations.
    - initial_U: Tensor containing the initial velocities.
    - dx: spacing in x direction.
    - dy: spacing in y direction.

    Returns:
    - Loss associated with the initial conditions.
    """
    # Filter the points inside the region
    mask_inside = ((initial_TXY[:, 1] >= 0.75) & (initial_TXY[:, 1] <= 1.25) & 
                   (initial_TXY[:, 2] >= 0.75) & (initial_TXY[:, 2] <= 1.25))
    
    mask_outside = ~mask_inside

    region_initial_TXY_inside = initial_TXY[mask_inside]
    region_initial_U_inside = initial_U[mask_inside]
    
    region_initial_TXY_outside = initial_TXY[mask_outside]
    region_initial_U_outside = initial_U[mask_outside]

    # Predict the values using neural network for the region inside
    predicted_U_region_inside = u_net(region_initial_TXY_inside)

    # Predict the values using neural network for the region outside
    predicted_U_region_outside = u_net(region_initial_TXY_outside)

    # Calculate the MSE loss for the values inside the region
    region_loss_inside = torch.mean((predicted_U_region_inside - region_initial_U_inside)**2)

    # Calculate the MSE loss for the values outside the region
    region_loss_outside = torch.mean((predicted_U_region_outside - region_initial_U_outside)**2)

    total_loss = region_loss_inside + region_loss_outside
    
    return total_loss


def initial_condition_loss_V2(u_net, initial_TXY):
    """
    Compute the loss associated with the initial condition.

    Args:
    - u_net: The neural network model.
    - initial_TXY: Tensor containing the initial time and spatial locations.

    Returns:
    - Loss associated with the initial conditions.
    """
    predicted_U = u_net(initial_TXY).squeeze().to(device)
    
    # Define the region where u should be 3
    mask_3 = (initial_TXY[:, 1] >= 0.75) & (initial_TXY[:, 1] <= 1.25) & (initial_TXY[:, 2] >= 0.75) & (initial_TXY[:, 2] <= 1.25)
    
    # Create a tensor of target values (either 3 or 1)
    target_U = torch.ones_like(predicted_U)  # default to 1
    target_U[mask_3] = 3  # set the inner region to 3
    
    loss = torch.mean((predicted_U - target_U)**2)
    return loss


# LOSS FUNCTIONS
def loss_func(u_net, TXY_pde, nu, c,  U_BC, TXY_IN, TXY_OUT, TXY_BOT, TXY_TOP):
  """
  Computes the total loss for the problem. The total loss is a sum of the loss of interior points of PDE
  and the boundary loss evaluation.
  
  Args:
    u_net (nn.Module): The neural network model.
    XY_pde (torch.tensor): The collocation points for the PDE.
    Re (float): The Reynolds number.
    U_inf (float): The free stream velocity.
    XY_IN (torch.tensor): The collocation points for the inlet.
    XY_OUT (torch.tensor): The collocation points for the outlet.
    XY_BOT (torch.tensor): The collocation points for the bottom wall.
    XY_TOP (torch.tensor): The collocation points for the top wall.
    XY_CIRC (torch.tensor): The collocation points for the boundary of the circle.
    
  Returns:
    torch.tensor: The total loss.
  """
  loss_pde = pde_loss(u_net, TXY_pde, nu, c) #loss interior points of PDE
  loss_bounds = boundaries_loss(u_net, TXY_IN, TXY_OUT, TXY_BOT, TXY_TOP, U_BC) # boundaries loss evaluation
  loss_total_PDE = loss_pde + loss_bounds
  return loss_total_PDE

def total_loss_func(u_net, loss_pde, TXY_data, U_data, selected_timesteps):
    """
    Computes the total hard loss. The hard loss is a sum of the weight of PDE loss and the mean squared error
    between the predicted and actual values for selected timesteps.
  
    Args:
      u_net (nn.Module): The neural network model.
      w_pde (float): The weight of the PDE loss.
      TXY_data (torch.tensor): The collocation points for the data.
      U_data (torch.tensor): The actual values at the collocation points.
      loss_pde (torch.tensor): The PDE loss.
      selected_timesteps (list or torch.tensor): The timesteps you're interested in.
  
    Returns:
      torch.tensor: The total hard loss.
    """
    total_residual = 0
    
    for t in selected_timesteps:
        current_mask = (TXY_data[:, 0] == t)
        current_TXY = TXY_data[current_mask]
        current_U_data = U_data[current_mask]
        res_data = u_net(current_TXY) - current_U_data
        total_residual += torch.mean(res_data**2)
        
    total_all_residuals =  total_residual / len(selected_timesteps)
    total_loss = total_all_residuals + loss_pde
    return total_loss
