# -*- coding: utf-8 -*-
"""

@author: Blaise Madiega
email : blaisemadiega@gmail.com


python main.py --data_path solution_u_implicit_data.csv
python main.py --data_path solution_u_implicit_data.csv --pretrained_model u_net_final.pth
"""

import torch
import numpy as np
from argparse import ArgumentParser
from pinn import *
from utilis import *
from plotting import *
import csv

import time

# Début du chronomètre
start_time = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # device

def is_multiple_of_10_minus_2(value, tolerance=1e-4):
    return abs(value * 100 - round(value * 100)) < tolerance

def truncate_two_decimals(value):
    return int(value * 100) / 100.0

def truncate_three_decimals(value):
    return int(value * 1000) / 1000.0

def main(args):
    file_path = args.data_path
    N_mes = 1000  # number of measurements points
    
    x_min, x_max = 0.0, 2.0
    y_min, y_max = 0.0, 2.0
    W_pde = 1.0
    dx = 0.01
    dy = 0.01
    
    dat = pd.read_csv(file_path)
    
    dat["temps"] = dat["temps"].apply(truncate_three_decimals)
    # Filtrer les données pour ne garder que celles dont le temps est un multiple de 10^-2
    dat = dat[dat["temps"].apply(is_multiple_of_10_minus_2)]
    # Étape 1: Identifiez et extrayez les valeurs u au timestep 0.003
    #u_at_0_003 = dat[dat["temps"] == 0.003]["u"].values

    # Étape 2: Remplacez les valeurs u aux timesteps 0, 0.001 et 0.002 par cette valeur
    #for t in [0, 0.001, 0.002]:
        #dat.loc[dat["temps"] == t, "u"] = u_at_0_003

    dat["x"] = dat["x"].apply(truncate_two_decimals)
    dat["y"] = dat["y"].apply(truncate_two_decimals)
    #dat["temps"] = dat["temps"].apply(truncate_three_decimals)

    T_dat= dat["temps"].values
    X_dat = dat["x"].values
    Y_dat = dat["y"].values
    u_dat = dat["u"].values #Velocity x direction
    unique_timesteps = dat['temps'].unique()
    selected_timesteps = unique_timesteps[:20]

    txy = []
    for i in range(0, X_dat.size):
        t = T_dat[i]
        x = X_dat[i]
        y = Y_dat[i]
        txy.append([t, x, y])
        
    txy = np.array(txy)
    TXY_data = torch.from_numpy(txy).float().to(device)

    u_data = []

    for i in range(0, u_dat.size):
        u = u_dat[i]
        u_data.append([u])
        
    u_data = np.array(u_data)
    U_data = torch.from_numpy(u_data).float().to(device)
    
    nu = 0.1 # 
    U_BC = 1.0
    c = 1.0 
    
    max_epochs = args.max_iter # maximum iterations

    #random_TXY_data, TXY_data, random_U_data, U_data, BC_TXY_data, BC_U_data = load_data(file_path, N_mes, device)
    
    # Initialize model
    u_net= PINN_NavierStokes(args.layers).to(args.device)
    # If a pretrained model is provided, load it. Otherwise, initialize the weights
    if args.pretrained_model is not None:
        try:
            u_net.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        except Exception as e:
            print(f"Error loading the pretrained model: {e}")
            print("Training the model from scratch...")
            u_net.apply(init_weights)
    else:
        u_net.apply(init_weights)
    
    # If we're not using a pretrained model, then we train the model
    if args.pretrained_model is None or args.pretrained_model is not None:
        # Optimisation
        optimizer = torch.optim.Adam(u_net.parameters(), lr = args.learning_rate)

        # Generate collocation points
        N_x = args.num_X_points
        N_y = args.num_Y_points
        N_bc = args.num_BC_points
        
        TXY_pde = colloc_pde_interior(N_x, N_y, x_min, x_max, y_min, y_max)
        TXY_in = colloc_Xinlet(N_bc, x_min, y_min, y_max)
        TXY_out = colloc_Xoutlet(N_bc, x_max, y_min, y_max)
        TXY_bot = colloc_Wall_bottom(N_bc, x_min, x_max, y_min)
        TXY_top = colloc_Wall_top(N_bc, x_min, x_max, y_max)
        
        # pde loss initialization
        # Extract initial condition data
        initial_indices = np.where(T_dat == np.min(T_dat))
        initial_TXY = TXY_data[initial_indices]
        initial_U = U_data[initial_indices]
        # Compute the initial condition loss
        loss_initial = initial_condition_loss_V2(u_net, initial_TXY)
        loss_pde = loss_initial + loss_func(u_net, TXY_pde, nu, c, U_BC, TXY_in, TXY_out, TXY_bot, TXY_top)
        total_loss = total_loss_func(u_net, loss_pde, TXY_data, U_data, selected_timesteps)

        train_losses_pde = []
        train_losses_total = []
        #val_losses = []
        epochs_dat = []
        # Create a CSV file to store the epoch and loss values
        with open('training_log_TC_05.csv', mode='w', newline='') as csvfile:
            log_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            log_writer.writerow(['Epoch', "Loss_PDE", 'Total_loss',  'Learning Rate'])  # Write the header row

            epoch = 0
            
        
            #while (epoch <= max_epochs and loss_pde.item() > 8.5e-7):
            while (epoch <= max_epochs and total_loss.item() > 8.5e-7):
                # loss for pde calculation
                loss_initial = initial_condition_loss_V2(u_net, initial_TXY)
                loss_pde = loss_initial + loss_func(u_net, TXY_pde, nu, c, U_BC, TXY_in, TXY_out, TXY_bot, TXY_top)
                total_loss = total_loss_func(u_net, loss_pde, TXY_data, U_data, selected_timesteps)
                
                train_losses_pde.append(loss_pde.item())
                train_losses_total.append(total_loss.item())
                epochs_dat.append(epoch)
                # backward
                total_loss.backward()
                #loss_pde.backward()
                # update
                optimizer.step()
                
                # Update the learning rate scheduler
                #scheduler.step(loss_hard)
                # Update the learning rate scheduler
                #learning_rate = exp_decay_schedule(epoch, initial_lr, decay_rate, plateau_epoch)
                #for param_group in optimizer.param_groups:
                    #param_group['lr'] = learning_rate

                # Write the epoch and loss values to the CSV file
                log_writer.writerow([epoch, loss_pde.item(), total_loss.item(), args.learning_rate])
                #log_writer.writerow([epoch, loss_pde.item(), loss_pde.item(), args.learning_rate])

                if epoch % 500 == 0:
                    #current_learning_rate = optimizer.param_groups[0]['lr']
                    print(f'epoch: {epoch}/{max_epochs}, loss_PDE: {loss_pde.item()}, total_loss:{total_loss.item()}, lr = {args.learning_rate}', flush=True)
                    #print(f'epoch: {epoch}, loss_PDE: {loss_pde.item()}, total_loss:{loss_pde.item()}, lr = {args.learning_rate}', flush=True)
                
                    # Save the U_NET model every 10,000 steps
                if epoch % 10000 == 0:
                    torch.save(u_net.state_dict(), f'u_net_epochTC_05_{epoch}.pth')
                
                optimizer.zero_grad()
                epoch += 1

            # Disable interactive mode at the end
            #plt.ioff()
            #plt.show()
    
        # Save your model
        PATH = "u_net_final_TC_05.pth"
        torch.save(u_net.state_dict(), PATH)

    X_pred = np.hstack((T_dat.flatten()[:,None], X_dat.flatten()[:,None], Y_dat.flatten()[:,None]))
    X_pred = torch.from_numpy(X_pred).float().to(device)

    # Deep Learning Model Prediction
    u_pred = u_net(X_pred)[:,0]
    U_pred = u_pred.detach().cpu().numpy().flatten() # Magnitude of Velocity predicted

    
    # Plotting results
    print("Results plotting ...")
    unique_times = np.unique(T_dat)
    max_errors = []
    #print(unique_times) # Obtenez tous les temps uniques
    for t in unique_times[20:]:
        frame_data = dat[dat["temps"] == t]
        X_f = frame_data['x'].to_numpy()
        Y_f = frame_data['y'].to_numpy()
        u_f = frame_data['u'].to_numpy()
        predicted_u = U_pred[np.where(T_dat == t)].squeeze()
        # Calcul de l'erreur absolue
        error = np.abs(u_f - predicted_u)
        max_errors.append(np.max(error))
        #plot_magnitude(X_f, Y_f, u_f, f'U at t={t} (ground truth)', f"pictures/U_solution_t{t}.png")
        # Utilisez la fonction plot_side_by_side pour afficher la comparaison
        plot_side_by_side(X_f, Y_f, u_f, predicted_u, error, t)
        #indices = np.where(T_dat == t)  # Obtenez tous les indices où le temps est égal à t
        #plot_magnitude(X_dat[indices].flatten(), Y_dat[indices].flatten(), U_pred[indices], f'U magnitude at t={t} (predicted with PINN)', f"pictures/U_magnitude_predicted_t{t}.png")

    average_max_error = np.mean(max_errors)
    FINAL_max_error = np.max(max_errors)
    print(f"Moyenne des erreurs maximales sur les pas de temps: {average_max_error}")
    print(f"MAXIMUM des erreurs maximales sur les pas de temps: {FINAL_max_error}")
    
    # 3. Afficher les pertes après l'entraînement
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_pde, label='PDE Loss')
    plt.plot(train_losses_total, label='Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('PDE and Total Losses over Epochs')
    plt.legend()
    plt.grid(False)

    # Mettre l'échelle en log sur l'axe des ordonnées
    plt.yscale('log')

    # 4. Enregistrer le graphique
    plt.savefig('losses_plot_tc_05.png')
    plt.show()
    
    end_time = time.time()

    execution_time = end_time - start_time

    # Convertir en heures, minutes, secondes
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = execution_time % 60

    # Écrire dans un fichier
    with open("temps_execution_TC_05.txt", "w") as file:
        file.write(f"Temps d'exécution: {hours} heure(s), {minutes} minute(s) et {seconds:.2f} seconde(s)\n")
        file.write(f"Moyenne des erreurs maximales sur les pas de temps: {average_max_error}, \n MAXIMUM des erreurs maximales sur les pas de temps: {FINAL_max_error} \n")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model configuration
    #parser.add_argument('--layers', type=int, nargs='+', default=[3, 64, 64, 64, 1])
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 72, 1])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Data configuration
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pretrained_model', type=str, help='Path to the pretrained model', default=None)
    parser.add_argument('--num_X_points', type=int, default=50)
    parser.add_argument('--num_Y_points', type=int, default=50)
    parser.add_argument('--num_BC_points', type=int, default=100)
    parser.add_argument('--max_iter', type=int, default=50000)
    args = parser.parse_args()

    main(args)