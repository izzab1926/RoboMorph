import torch 
import matplotlib.pyplot as plt 
import numpy as np

def chirp_signal(n_steps,attenuation_factor,frequency,j):

    time_steps = torch.linspace(0, n_steps, n_steps)
    t = time_steps.unsqueeze(1)
    t = t/60

    phi = torch.rand(1).uniform_(-np.pi,np.pi)
    q0 = torch.rand(1).uniform_(-.5, .5) 
    a = torch.rand(1).uniform_(-4,4)    #  [ -3,3]  

    if frequency < 0.3:
        f1 = torch.rand(1).uniform_(frequency/1.1,frequency*1.5)
        f2 = torch.rand(1).uniform_(frequency/1.5, frequency*2)
    else:
        f1 = torch.rand(1).uniform_(frequency/1.3,frequency/1.2)
        f2 = torch.rand(1).uniform_(frequency/1.1, frequency*1.1)
    

    if j>=7:
        _trajectory = 0*t
    else:
        _trajectory = q0 + torch.sign(torch.rand(1).uniform_(-1,1))* a * torch.cos (2* np.pi * f1 *( 1 + 1/4 * torch.cos(  2 * np.pi * f2* t))*t + phi)

    trajectory = torch.Tensor(_trajectory)
    trajectory = trajectory.view(n_steps) 
    return trajectory



def control_action_chirp(n_simulations , n_steps , n_dofs, frequency):
    
    # Function to generate whole bunchs of simulations
    
    all_trajectories = torch.zeros(n_simulations,n_dofs,n_steps)

    for i in range(n_simulations):
        for j in range(n_dofs):
            if j ==8 or j==7:
                attenuation_factor=np.inf
            else: 
                attenuation_factor = 2 # 1.6 safe

            single_dof = chirp_signal(n_steps,attenuation_factor,frequency,j)
            all_trajectories[i][j] = single_dof

    # print(all_trajectories.shape) #n_simulations x n_dofs x n_steps 

    return all_trajectories


if __name__ == '__main__':

    torch.manual_seed(10)  
    for i in range(1):
        n_dofs = 4
        n_steps = 1000
        n_simulations = 10
        frequency = 0.3
        time_steps = torch.linspace(0, n_steps, n_steps )

        my_control_action = control_action_chirp(n_simulations , n_steps , n_dofs, frequency)

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

