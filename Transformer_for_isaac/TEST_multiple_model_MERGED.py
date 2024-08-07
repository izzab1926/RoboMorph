from pathlib import Path
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_sim import Config, TSTransformer
import pandas as pd
from extrapolation import check_ID_OOD
import metrics
from torch.utils.data import random_split
import os   
import argparse           
from tabulate import tabulate

torch.manual_seed(420)
np.random.seed(430)
torch.manual_seed(420)
np.random.seed(430)

parser = argparse.ArgumentParser(description='RoboMorph')

# Overall Parser Parameters 

parser.add_argument('--skip', type=int, default=0, metavar='num',
                    help='number of timesteps to skip')
parser.add_argument('--test-folder', type=str, default="train", metavar='test',
                    help='test-folder')
parser.add_argument('--figure-folder', type=str, default="fig_prova", metavar='figure',
                    help='Figure output')
parser.add_argument('--model-folder', type=str, default="out_ds_big", metavar='model',
                    help='Figure output')
parser.add_argument('--save-figure', action='store_true', default=True,
                    help='Figure output')
parser.add_argument('--plot-predictions-and-metrics', action='store_true', default=True,
                    help='Figure output')
parser.add_argument('--num-test', type=int, default=1000, metavar='num',
                    help='number of wanted test trajectories for each file')

test_cfg,unparsed = parser.parse_known_args() 

# Overall settings

fig_path = Path(f"{test_cfg.figure_folder}")
fig_path.mkdir(exist_ok=True)

log = open(fig_path / "log.txt","w")
out_dir = f"{test_cfg.model_folder}"

save_figure = test_cfg.save_figure
plot_predictions_and_metrics = test_cfg.plot_predictions_and_metrics

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 20}
plt.rc('font', **font)

# Compute settings
cuda_device = "cuda:0"
no_cuda = False
threads = 10
compile = False 

# Configure compute
torch.set_num_threads(threads) 
use_cuda = not no_cuda and torch.cuda.is_available()
device_name  = cuda_device if use_cuda else "cpu"
device = torch.device(device_name)
device_type = 'cuda' if 'cuda' in device_name else 'cpu' # for later use in torch.autocast
#torch.set_float32_matmul_precision("highest") 
#torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


# skip is a variable that allow you to skip the first timesteps of a given file
skip = test_cfg.skip
print("Skipping the first: ", skip, "timesteps")

# ---------------------------- Testing FOLDER ----------------------------------------------

parent_folder = os.path.join(os.getcwd(), os.pardir) 
parent_folder = os.path.abspath(parent_folder)

try:
    relative_folder = f"./data_generation/out_tensors/{test_cfg.test_folder}"
    tensors_path = os.path.join(parent_folder,relative_folder ) 
    tensors_path = os.path.abspath(tensors_path)
    list_of_available_tensors = os.listdir(tensors_path)

except FileNotFoundError:
    relative_folder = f"Data_generation/python/examples/Franka/out_tensors/{test_cfg.test_folder}"
    tensors_path = os.path.join(parent_folder,relative_folder ) 
    tensors_path = os.path.abspath(tensors_path)
    list_of_available_tensors = os.listdir(tensors_path)

print("\nThese are the available test dataset in "+str(tensors_path) +":\n")
log.write("\n These are the available test dataset in "+str(tensors_path) +":\n")
for i in range(len(list_of_available_tensors)):
    single_pt_file_path = os.path.join(tensors_path, list_of_available_tensors[i]) 
    print("--> ",list_of_available_tensors[i])
    log.write("--> " + list_of_available_tensors[i] + "\n")

# --------------------------- MODEL FOLDER  --------------------------------------
model_path = os.path.abspath(f"./{test_cfg.model_folder}")
list_of_available_models = os.listdir(model_path)

print("\nThese are the available models in "+str(model_path) +":\n")
log.write("\nThese are the available models in "+str(model_path) +":\n")

for i in range(len(list_of_available_models)):
    single_model_file_path = os.path.join(model_path, list_of_available_models[i]) 
    print("--> ",list_of_available_models[i])
    log.write("--> " + list_of_available_models[i]+ "\n")

# ---------------------------------------------------------------------------------
    
test_metrics = torch.zeros((len(list_of_available_models),len(list_of_available_tensors),8))

for model_idx in range(len(list_of_available_models)):

    first_idx = list_of_available_models[model_idx].find("_ds")
    last_idx = list_of_available_models[model_idx].find(".pt")
    dataset_name = list_of_available_models[model_idx][first_idx:last_idx]
    training_list_file = f"training{dataset_name}_list.txt"        
    
    f = open(fig_path / f"merged_{list_of_available_models[model_idx][:last_idx]}_{test_cfg.test_folder}.txt", "w")
    f.write(f"\Model --> { list_of_available_models[model_idx]} \n")

    model_name = list_of_available_models[model_idx]
    print(" \n --> Testing model: ", model_name )
    log.write("\n\n *********** Testing model: "+ model_name + ' ************ \n')
    
    # load model's wieghts (partition_**.pt file)
    out_dir = Path(out_dir) # "out"
    exp_data = torch.load(out_dir/model_name, map_location="cpu")
    cfg = exp_data["cfg"]

    # It is possible to test on different scales, even it was trained on a different partition 
    # Just overwrite
    # cfg.seq_len_ctx = 500
    # cfg.seq_len_new = 150
    
    seq_len = cfg.seq_len_ctx + cfg.seq_len_new

    model_args = exp_data["model_args"]
    conf = Config(**model_args)
    model = TSTransformer(conf).to(device)
    model.load_state_dict(exp_data["model"]);
    print(f'\ntrain_time: {round(exp_data["train_time"] / 60,4)} minutes')
    log.write(f'\ntrain_time: {round(exp_data["train_time"] / 60,4)} minutes')
    # test all the available files 
    # Empty list for boxplot
    list_of_R2 = []
    list_of_RMSE = []

    for test_idx in range(len(list_of_available_tensors)): 

        single_pt_file_path = os.path.join(tensors_path, list_of_available_tensors[test_idx]) 

        log.write("\nTEST DATASET: " + list_of_available_tensors[test_idx])

        loaded = torch.load(single_pt_file_path,map_location=device) 
        
        # Loading single test file: if it a number of timesteps equal to 501 or 1001, remove the first.
        # The first time step of contro action is set for everyone to 0, and it is in the generation.

        @ torch.no_grad()
        def loading():
            if loaded['control_action'].shape[0] == seq_len + 1:
                control_action_extracted = loaded['control_action'][skip + 1:,:test_cfg.num_test,:7] #time x num_envs x dofs
            else:
                control_action_extracted = loaded['control_action'][skip:,:test_cfg.num_test,:]
            position_extracted = loaded['position'][skip:,:test_cfg.num_test,:]

            position = torch.movedim(position_extracted.to('cpu'),-2,-3)
            control_action = torch.movedim(control_action_extracted.to('cpu'),-2,-3)
            return control_action,position
        
        control_action,position = loading()
        num_test = control_action.shape[0] 
        ny = position.shape[2] 

        if test_idx == 0 and model_idx ==0:
            #This is the setting of random_idx for everyone!
            random_idx = int(np.ceil(torch.rand(1).uniform_(0,num_test-1).numpy()))
            print("Number of tests: ", num_test, "random idx: ", random_idx)

        # Initializing folders
        model_path = model_name.replace(".pt"," ")
        model_path = Path(fig_path / f"{model_path}")
        model_path.mkdir(exist_ok=True)
        test_path = list_of_available_tensors[test_idx].replace(".pt"," ")
        test_path = Path(model_path / f"{test_path}")
        test_path.mkdir(exist_ok=True)

        # This check, consider the the actual test respect the training Dataset.

        training,test,output,boolean_output = check_ID_OOD(list_of_available_tensors[test_idx], training_list_file)

        if any(map(lambda elem: elem is None, (training,test,output,boolean_output))):
            output = "Error_validation"
        else:
            differing_classes = boolean_output.count(False)

        if  output == "Error_validation":
            suptitle = "\nValidation"
        elif output == "In-Distribution":
            suptitle = f"\nID\n{test['task']}"
        elif output == "Out-Of-Distribution":
            suptitle = f"\nOOD ({differing_classes} cat)\n{test['task']} "    

        #------------------------- initializing errors ---------------------------------
        
        y_ground_truth = torch.empty(position[:num_test,cfg.seq_len_ctx:seq_len,:].shape,device=device)
        y_predicted = torch.empty(position[:num_test,cfg.seq_len_ctx:seq_len,:].shape,device=device)
        sim_error = torch.empty(position[:num_test,cfg.seq_len_ctx:seq_len,:].shape,device=device)

        for i in range(control_action.shape[0]): 

            single_u = control_action[i,:seq_len,:]
            single_y = position[i,:seq_len,:]
            single_y = single_y.to(device)
            single_u = single_u.to(device)

            mean_u = single_u.mean(axis=1, keepdim=True)
            std_u = single_u.std(axis=1, keepdim=True)

            # Normalizza il tensore
            single_u = (single_u - mean_u) / (std_u + 1e-6)

            mean_y = single_y.mean(axis=1, keepdim=True)
            std_y = single_y.std(axis=1, keepdim=True)

            # Normalizza il tensore
            single_y = (single_y - mean_y) / (std_y + 1e-6)

            with torch.no_grad():
                single_y_ctx = single_y[:cfg.seq_len_ctx, :]
                single_u_ctx = single_u[:cfg.seq_len_ctx, :]
                single_y_new = single_y[cfg.seq_len_ctx:seq_len, :]
                single_u_new = single_u[cfg.seq_len_ctx:seq_len, :]
                # print(single_y_ctx.shape)
                t_start = time.time()
                single_y_sim = model(single_y_ctx.unsqueeze(0), single_u_ctx.unsqueeze(0), single_u_new.unsqueeze(0))

                
                t_end = time.time()
                single_sim_error = single_y_sim - single_y_new

            # -------------------------------------------------------------------------
            y_ground_truth[i] = single_y_new
            y_predicted[i] = single_y_sim
            sim_error[i] = single_sim_error

        # ------------------------------- Metrics calculation -----------------------------
        
        t = np.arange(1, single_u_new.shape[0]+1) + cfg.seq_len_ctx
        y_ground_truth = y_ground_truth.to("cpu").detach().numpy()
        y_predicted = y_predicted.to("cpu").detach().numpy()
        sim_error = sim_error.to("cpu").detach().numpy()
        
        # imposing here ny different from dimension of input, means actually consider the metrics only of the first 4
        # ny = 3

        y_ground_truth=y_ground_truth[:,:,:ny]
        y_predicted =y_predicted[:,:,:ny]
        sim_error = sim_error[:,:,:ny]

        RMSE = metrics.rmse(y_ground_truth, y_predicted, time_axis=1)
        MSE = RMSE **2
        print(RMSE.shape)
        NRMSE = metrics.nrmse(y_ground_truth, y_predicted, time_axis=1)
        fit_index = metrics.fit_index(y_ground_truth, y_predicted, time_axis=1)
        r_squared = metrics.r_squared(y_ground_truth, y_predicted, time_axis=1)

        # reshaping into ny different means
        y_ground_truth_reshaped = y_ground_truth.reshape(-1,ny)
        y_predicted_reshaped = y_predicted.reshape(-1,ny)

        print(y_ground_truth.shape)
        
        RMSE_reshaped = metrics.rmse(y_ground_truth_reshaped, y_predicted_reshaped, time_axis=0).reshape(14,1)
        NRMSE_reshaped = metrics.nrmse(y_ground_truth_reshaped, y_predicted_reshaped, time_axis=0).reshape(14,1)
        fit_index_reshaped = metrics.fit_index(y_ground_truth_reshaped, y_predicted_reshaped, time_axis=0).reshape(14,1)
        r_squared_reshaped = metrics.r_squared(y_ground_truth_reshaped, y_predicted_reshaped, time_axis=0).reshape(14,1)

        # Define the metrics and quantities
        metrics_labels = ["R^2","RMSE", "NRMSE", "Fit Index"]
        quantities = ["x","y","z" ,"X","Y","Z" ,"W","q0","q1","q2","q3","q4","q5","q6"]

        # Create a DataFrame
        data = {
            metrics_labels[0]: np.around(r_squared_reshaped.ravel(),3), 
            metrics_labels[1]: np.around(RMSE_reshaped.ravel(),4),       
            metrics_labels[2]: np.around(NRMSE_reshaped.ravel(),4),  
            metrics_labels[3]: np.around(fit_index_reshaped.ravel(),2)}

        df = pd.DataFrame(data, index=quantities)

        # Display the table using tabulate

        print(f"\nTest --> { list_of_available_tensors[test_idx]} \n")
        print(tabulate(df, headers='keys', tablefmt='grid'))
        f.write(f"\n\nTest --> { list_of_available_tensors[test_idx]} \n")
        f.write(f"{suptitle}\n")
        f.write(tabulate(df, headers='keys', tablefmt='grid'))
        
        labels_coordinates = ["$x$","$y$","$z$" ,"$X$","$Y$","$Z$" ,"$W$","q0","q1","q2","q3","q4","q5","q6"]
        labels_pred = ["$\hat x$","$\hat y$","$\hat z$","$\hat X$","$ \hat Y$","$\hat Z$" ,"$ \hat W$"
                    ,"$ \hat q0$","$ \hat q1$","$ \hat q2$","$ \hat q3$","$ \hat q4$","$ \hat q5$","$ \hat q6$"]
        labels_error = ["$x - \hat x$","$y - \hat y$","$z - \hat z$","$X - \hat X$","$ Y - \hat Y$",
                        "$Z - \hat Z$" ,"$ W -  \hat W$","$ q0 -  \hat q0$","$ q1 -  \hat q1$","$ q2 -  \hat q2$"
                        ,"$ q3 -  \hat q3$","$ q4 -  \hat q4$","$ q5 -  \hat q5$","$ q6 -  \hat q6$"] 

        # ------------------------- Print MERGED TRAJECTORIES  -----------------------------------------

        font = {'family' : 'DejaVu Sans',
                'weight' : 'normal',
                'size'   : 20}
        plt.rc('font', **font)

        t_context = np.arange(1, cfg.seq_len_ctx) 
        t_prediction = np.arange(1, single_u_new.shape[0]+1) + cfg.seq_len_ctx
        t_total = np.arange(0, position.shape[1])

        fig, axs = plt.subplots(round(ny/2), 2,figsize=(20, 20))
        fig.suptitle('Prediction of merged trajectories ')
        k = 0
        show = 12000

        for j in range(int(2)):
            for i in range(round(ny/2)):
                axs[i,j].plot(y_ground_truth_reshaped[:show, k], 'green',linewidth=1,label=labels_pred[k])
                axs[i,j].plot(y_predicted_reshaped[:show, k], 'magenta',linewidth=1,label=labels_coordinates[k])
                # Plot entire Signal
                axs[i,j].grid(True)
                axs[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                for step in range(0, show, 800):
                    axs[i,j].axvline(x=step, color='gray', linestyle='--', linewidth=1, alpha=0.8)
                    if k ==0:
                        axs[i,j].text(step + 380, 1.2* np.max(y_ground_truth_reshaped[:show, k]), 
                                      f'{int(np.round(step/800))}', color='black', fontsize=15,
                                        bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none',
                                                   boxstyle='round,pad=0.3'), fontweight='bold')
                    if k==7 :
                        axs[i,j].text(step + 380, 1.4* np.max(y_ground_truth_reshaped[:show, k]), 
                                      f'{int(np.round(step/800))}', color='black', fontsize=15, 
                                      bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none', 
                                                boxstyle='round,pad=0.3'), fontweight='bold')
                if k!=6 and k!=13:
                    axs[i,j].set_xticks([])
                k=k+1
            fig.tight_layout()    
        if save_figure == True:
            plt.savefig(test_path / f"merged_trajectories.png")
            plt.close()
        else:
            plt.show()

        #R2
        r_squared_mean_along_y_dimensions = torch.empty((ny,1),device = device)
        r_squared_std_along_y_dimensions = torch.empty((ny,1),device = device)
        #RMSE
        RMSE_mean_along_y_dimensions = torch.empty((ny,1),device = device)
        RMSE_std_along_y_dimensions = torch.empty((ny,1),device = device)
        # NRMSE
        NRMSE_mean_along_y_dimensions = torch.empty((ny,1),device = device)
        NRMSE_std_along_y_dimensions = torch.empty((ny,1),device = device)
        # fit index
        fit_index_mean_along_y_dimensions = torch.empty((ny,1),device = device)
        fit_index_std_along_y_dimensions = torch.empty((ny,1),device = device)

        for i in range(r_squared.shape[1]):
            r_squared_mean_along_y_dimensions[i]= torch.tensor(r_squared[:,i].mean())
            r_squared_std_along_y_dimensions[i] = torch.tensor(r_squared[:,i].std())
            RMSE_mean_along_y_dimensions[i]= torch.tensor(RMSE[:,i].mean())
            RMSE_std_along_y_dimensions[i] = torch.tensor(RMSE[:,i].std())
            NRMSE_mean_along_y_dimensions[i]= torch.tensor(NRMSE[:,i].mean())
            NRMSE_std_along_y_dimensions[i] = torch.tensor(NRMSE[:,i].std())
            fit_index_mean_along_y_dimensions[i]= torch.tensor(fit_index[:,i].mean())
            fit_index_std_along_y_dimensions[i] = torch.tensor(fit_index[:,i].std())

        metrics_values_EE = [RMSE[:,:7],r_squared[:,:7],NRMSE[:,:7],fit_index[:,:7]]
        metrics_values_joint_pos = [RMSE[:,7:],r_squared[:,7:],NRMSE[:,7:],fit_index[:,7:]]

        font = {'family' : 'DejaVu Sans',
                'weight' : 'normal',
                'size'   : 15}
        plt.rc('font', **font)

        # ------------------------ Variability coordinate by coordinate ------------------------

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18,18))
        fig.suptitle(f'Metrics EE - model: {model_name} \n test {list_of_available_tensors [test_idx]}')

        for j in range(4):
            # Plot boxplots for each x-coordinate in the same subplot
            axes[j].boxplot(metrics_values_EE[j],1,'',patch_artist = True, 
                            boxprops = dict(facecolor = "lightblue"), 
                            medianprops = dict(color = "green", linewidth = 1.5), 
                            whiskerprops = dict(color = "red", linewidth = 2), 
                            tick_labels=[f"{quantities[i]}" for i in range(7)])
            axes[j].set_title(f'${metrics_labels[j]}$')
            axes[j].grid(alpha=0.5)

        if save_figure == True:
            plt.savefig(test_path / f"metrics_EE.png")
            plt.close()
        else:
            plt.show()

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18,18))
        fig.suptitle(f'Metrics joint positions - model: {model_name} \n test {list_of_available_tensors [test_idx]}')

        for j in range(4):
            # Plot boxplots for each x-coordinate in the same subplot
            axes[j].boxplot(metrics_values_joint_pos[j],1,'',patch_artist = True, 
                            boxprops = dict(facecolor = "lightblue"), 
                            medianprops = dict(color = "green", linewidth = 1.5), 
                            whiskerprops = dict(color = "red", linewidth = 2), 
                            tick_labels=[f"{quantities[i+7]}" for i in range(7)])
            
            axes[j].set_title(f'${metrics_labels[j]}$')
            # axes[j].set_ylabel(f"${metrics_labels[j]}")
            axes[j].grid(alpha=0.5)

        # Show the plot
        
        if save_figure == True:
            plt.savefig(test_path / f"metrics_joint_pos.png")
            plt.close()
        else:
            plt.show()
f.close()

torch.cuda.empty_cache()

