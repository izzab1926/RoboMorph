import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_control(n_steps,frequency,attenuation_factor,j):

    time_steps = torch.linspace(0, n_steps, n_steps ) 
    # Randomly generate coefficients for a smooth trajectory
    r1,r2 = -1,1
    a = 2* torch.rand(4).uniform_(-frequency*15,frequency *15)    #  [ -3,3]    

    t = time_steps.unsqueeze(1)

    t = t * 1/60 # Simulation dt and this t must be the same.
    freq = 2 * np.pi * torch.rand(1).uniform_(frequency/1.5, frequency*1.5) #(frequency/10, frequency*1.5)
    _trajectory =  torch.sign(torch.rand(1).uniform_(r1,r2)) * (a[0]*torch.sin(freq*t) 
                         + a[1] * torch.cos(freq*1.5*t) + a[2] *torch.sin(freq*2*t) 
                         +torch.sign(torch.rand(1).uniform_(r1,r2))* a[3] * torch.cos(freq*3*t))/attenuation_factor 
    
    trajectory = torch.Tensor(_trajectory)
    trajectory = trajectory.view(n_steps) 
    return trajectory


def control_action_function( n_simulations , n_steps , n_dofs, frequency, *args):

    lower_args = [arg.lower() for arg in args]

    for i in args:
        if args == None:
            print("Franka")
        elif args == "Kuka":
            print("Kuka")
    # Function to generate whole bunchs of simulations
    all_trajectories = torch.zeros(n_simulations,n_dofs,n_steps)
    for i in range(n_simulations):

        for j in range(n_dofs):
            if j ==8 or j==7:
               attenuation_factor = np.inf
            elif j == 1:
                attenuation_factor = 1.3 # 1.3 # 2 for real
                if "kuka" in lower_args:
                    attenuation_factor = 20
            else:
                attenuation_factor = 1.6 # 1.6 # 3 for real 
                if "kuka" in lower_args:
                    attenuation_factor = 25 
                    
#  Here you can freeze all joints unless joint 1 
        # for j in range(n_dofs):
        #     if j == 1:
        #         attenuation_factor = 2
        #     else:
        #         attenuation_factor = np.inf

            
            single_dof = generate_control(n_steps,frequency,attenuation_factor,j)
            all_trajectories[i][j] = single_dof
    return all_trajectories     #n_simulations x n_dofs x n_steps 




# -----------------------------------------------------------------------------------------------

if __name__ == '__main__':


    torch.manual_seed(123)  
    
    n_dofs = 9
    n_steps = 1000
    n_simulations = 3
    frequency = .2
    time_steps = torch.linspace(0, n_steps, n_steps )

    # no additional argument --> Franka
    # my_control_action = control_action_function(n_simulations , n_steps , n_dofs, frequency)

    # no additional argument --> Franka
    my_control_action = control_action_function(n_simulations , n_steps , n_dofs, frequency,"kuka")

    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 9})
    for i in range(n_simulations):
        for j in range(n_dofs):
            plt.subplot(n_simulations, n_dofs, i * n_dofs + j + 1)
            for k in range(n_dofs):
                plt.grid(color='k', linestyle='-', linewidth=0.2)
                plt.plot(time_steps, my_control_action[:][i][j].numpy(), alpha=1, linewidth=0.8)
            plt.title(f"Simulation {i+1}, DOF {j+1}")
            plt.xlabel("Time Steps")
            plt.ylabel("Position")
    plt.tight_layout()
    plt.show()


