from pathlib import Path
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformer_sim import Config, TSTransformer
#import tqdm
import argparse
import metrics
from torch.utils.data import random_split
import os   

fig_path = Path("fig")
fig_path.mkdir(exist_ok=True)

log = open(fig_path / "log.txt","w")

torch.manual_seed(420)
np.random.seed(430)
torch.manual_seed(420)
np.random.seed(430)

# Overall settings
out_dir = "out"
save_figure = True
# System settings
nu = 7
ny = 14

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

torch.set_float32_matmul_precision("high") 
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# ---------------------------- Testing file ----------------------------

parent_folder = os.path.join(os.getcwd(), os.pardir) 
parent_folder = os.path.abspath(parent_folder)

try:
    relative_folder = "isaacgym/python/examples/Franka/out_tensors/test"
    tensors_path = os.path.join(parent_folder,relative_folder ) 
    tensors_path = os.path.abspath(tensors_path)
    list_of_available_tensors = os.listdir(tensors_path)

except FileNotFoundError:
    relative_folder = "Data_generation/python/examples/Franka/out_tensors/test"
    tensors_path = os.path.join(parent_folder,relative_folder ) 
    tensors_path = os.path.abspath(tensors_path)
    list_of_available_tensors = os.listdir(tensors_path)

print("\nThese are the available test dataset in "+str(tensors_path) +":\n")
log.write("\n These are the available test dataset in "+str(tensors_path) +":\n")
for i in range(len(list_of_available_tensors)):
    single_pt_file_path = os.path.join(tensors_path, list_of_available_tensors[i]) 
    print("--> ",list_of_available_tensors[i])
    log.write("--> " + list_of_available_tensors[i] + "\n")

# --------------------------- Testing model --------------------------------------
model_path = os.path.abspath("./out")
list_of_available_models = os.listdir(model_path)
print("\nThese are the available models in "+str(model_path) +":\n")
log.write("\nThese are the available models in "+str(model_path) +":\n")
for i in range(len(list_of_available_models)):
    single_model_file_path = os.path.join(model_path, list_of_available_models[i]) 
    print("--> ",list_of_available_models[i])
    log.write("--> " + list_of_available_models[i]+ "\n")
# ---------------------------------------------------------------------------------
    
test_metrics = torch.zeros((len(list_of_available_models),len(list_of_available_tensors),4))

for model_idx in range(len(list_of_available_models)):

    model_name = list_of_available_models[model_idx]
    print(" \n --> Testing model: ", model_name )
    log.write("\n\n *********** Testing model: "+ model_name + ' ************ \n')
    
    # load model's wieghts (partition_**.pt file)
    out_dir = Path(out_dir) # "out"
    exp_data = torch.load(out_dir/model_name, map_location="cpu")
    cfg = exp_data["cfg"]

    # Is it possible to test on different scales, even it was trained on a different partition 
    # cfg.seq_len_ctx = 500
    
    # cfg.seq_len_new = 150
    
    seq_len = cfg.seq_len_ctx + cfg.seq_len_new

    model_args = exp_data["model_args"]
    conf = Config(**model_args)
    model = TSTransformer(conf).to(device)
    model.load_state_dict(exp_data["model"]);

    # test all the available files 

    for test_idx in range(len(list_of_available_tensors)): 

        single_pt_file_path = os.path.join(tensors_path, list_of_available_tensors[test_idx]) 

        # print("\n TEST DATASET: ", list_of_available_tensors[test_idx])

        log.write("\nTEST DATASET: " + list_of_available_tensors[test_idx])

        loaded = torch.load(single_pt_file_path,map_location=device) #

        # Import All the 1000 steps, eventhough prediction may be on a subset
        
        @ torch.no_grad()
        def loading():
            control_action_extracted = loaded['control_action']
            position_extracted = loaded['position']
            position = torch.movedim(position_extracted.to('cpu'),-2,-3)
            control_action = torch.movedim(control_action_extracted.to('cpu'),-2,-3)
            return control_action,position
        
        @ torch.no_grad()
        def loading_OSC():
            control_action_extracted = loaded['control_action']
            position_extracted = loaded['position']
            position = torch.movedim(position_extracted.to('cpu'),-2,-3)
            # control_action = torch.movedim(control_action_extracted[:,:7,:-1].to('cpu'),-2,-1) 
            control_action = control_action_extracted.movedim(-1,-2)[:,:,:7]
            control_action = control_action[:,:1000,:]
            return control_action,position

        # control_action,position = loading_OSC()
        control_action,position = loading()

        num_test_robots = control_action.shape[0] 
        
        #------------------------- initializing errors ---------------------------------
        
        # y_ground_truth = torch.empty(position[:,cfg.seq_len_ctx:,:].shape,device=device)
        # y_predicted = torch.empty(position[:,cfg.seq_len_ctx:,:].shape,device=device)
        # sim_error = torch.empty(position[:,cfg.seq_len_ctx:,:].shape,device=device)
       
        y_ground_truth = torch.empty(position[:,cfg.seq_len_ctx:seq_len,:].shape,device=device)
        y_predicted = torch.empty(position[:,cfg.seq_len_ctx:seq_len,:].shape,device=device)
        sim_error = torch.empty(position[:,cfg.seq_len_ctx:seq_len,:].shape,device=device)

        # print(" \n------------- Calculating errors... ----------------- ")

        # for k in range(num_test_robots): 

        #     single_u = control_action[k,:,:].to(device)
        #     single_y = position[k,:,:].to(device)

        #     with torch.no_grad():
        #         single_y_ctx = single_y[:cfg.seq_len_ctx, :]
        #         single_u_ctx = single_u[:cfg.seq_len_ctx, :]
        #         single_y_new = single_y[cfg.seq_len_ctx:, :]
        #         single_u_new = single_u[cfg.seq_len_ctx:, :]

        #         single_y_sim = model(single_y_ctx.unsqueeze(0), single_u_ctx.unsqueeze(0), single_u_new.unsqueeze(0))
        #         single_sim_error = single_y_sim - single_y_new
            
        #     y_ground_truth[k] = single_y_new
        #     y_predicted[k] = single_y_sim
        #     sim_error[k] = single_sim_error

        # print(f"\nDone.")

        for i in range(control_action.shape[0]): 

            single_u = control_action[i,:seq_len,:]
            single_y = position[i,:seq_len,:]

            single_y = single_y.to(device)
            single_u = single_u.to(device)

            with torch.no_grad():
                single_y_ctx = single_y[:cfg.seq_len_ctx, :]
                single_u_ctx = single_u[:cfg.seq_len_ctx, :]
                single_y_new = single_y[cfg.seq_len_ctx:seq_len, :]
                single_u_new = single_u[cfg.seq_len_ctx:seq_len, :]
                t_start = time.time()
                single_y_sim = model(single_y_ctx.unsqueeze(0), single_u_ctx.unsqueeze(0), single_u_new.unsqueeze(0))
                t_end = time.time()
                single_sim_error = single_y_sim - single_y_new
            
            y_ground_truth[i] = single_y_new
            y_predicted[i] = single_y_sim
            sim_error[i] = single_sim_error

        # -----------------------------------  WIP --------------------------------- 
        
        training_frequency = 0.15 #extract from the file !!! WIP 
        training_frequency = str(training_frequency).replace('.','_')
        mass_bounds = 10,10
        mass_bounds = str(mass_bounds).replace('(','').replace(')','').replace(', ','_')

        current_test = list_of_available_tensors[test_idx]
        frequency_index = current_test.find('f_')
        randomization_index = current_test.find('mass_') 
        
        # --------------------------------------------------------------------------
        if current_test[frequency_index+2:frequency_index+6] == str(training_frequency):
            suptitle1 = 'Same frequency ('+ str(training_frequency).replace('_','.') + ' Hz) '
        else:
            suptitle1 = 'Different Frequency ( trained on: '+ str(training_frequency).replace('_','.') + ' vs ' + current_test[frequency_index+2:frequency_index+5].replace('_','.')+' Hz ) -- '

        if current_test[randomization_index+5:randomization_index+10].replace('p','') == mass_bounds:
            suptitle2 = 'Same mass bounds (+'+ current_test[randomization_index+5:randomization_index+10].replace('_','% -') +'%)'
        else:
            suptitle2 = 'Different bounds (trained on: +' + mass_bounds.replace('_',' -')+'% vs +'+  current_test[randomization_index+5:randomization_index+10].replace('_',' -').replace('.p','') +' %) '

        suptitle = suptitle1 + suptitle2 

        # -------------------------------Metrics calculation -----------------------------
        t = np.arange(1, single_u_new.shape[0]+1) + cfg.seq_len_ctx

        y_ground_truth = y_ground_truth.to("cpu").detach().numpy()
        y_predicted = y_predicted.to("cpu").detach().numpy()
        sim_error = sim_error.to("cpu").detach().numpy()

        
        # nrmse = metrics.nrmse(y_ground_truth, y_predicted, time_axis=1)
        # fit_index = metrics.fit_index(y_ground_truth, y_predicted, time_axis=1)

        rmse = metrics.rmse(y_ground_truth, y_predicted, time_axis=1)
        r_squared = metrics.r_squared(y_ground_truth, y_predicted, time_axis=1)

        r_squared_mean_along_y_dimensions = torch.empty((num_test_robots,1),device = device)
        r_squared_var_along_y_dimensions = torch.empty((num_test_robots,1),device = device)
        rmse_mean_along_y_dimensions = torch.empty((num_test_robots,1),device = device)
        rmse_var_along_y_dimensions = torch.empty((num_test_robots,1),device = device)

        for i in range(r_squared.shape[0]):

            r_squared_mean_along_y_dimensions[i]= torch.tensor(r_squared[i,:].mean())
            r_squared_var_along_y_dimensions[i] = torch.tensor(r_squared[i,:].var())
            rmse_mean_along_y_dimensions[i]= torch.tensor(rmse[i,:].mean())
            rmse_var_along_y_dimensions[i] = torch.tensor(rmse[i,:].var())

        # print(r_squared_mean_along_y_dimensions.shape) # overall mean of R^2 for the dimensions (mean along the 14 output dimensions) 

        r_squared_mean_along_y_dimensions = r_squared_mean_along_y_dimensions.to("cpu").detach().numpy()
        r_squared_var_along_y_dimensions = r_squared_var_along_y_dimensions.to("cpu").detach().numpy()
        rmse_mean_along_y_dimensions = rmse_mean_along_y_dimensions.to("cpu").detach().numpy()
        rmse_var_along_y_dimensions = rmse_var_along_y_dimensions.to("cpu").detach().numpy()

        r_squared_mean_along_batches = np.ones(num_test_robots)*r_squared_mean_along_y_dimensions.mean()
        r_squared_var_along_batches = np.ones(num_test_robots)*r_squared_var_along_y_dimensions.mean()
        rmse_mean_along_batches = np.ones(num_test_robots)*rmse_mean_along_y_dimensions.mean()
        rmse_var_along_batches = np.ones(num_test_robots)*rmse_var_along_y_dimensions.mean()

        # WIP 
        test_metrics[model_idx,test_idx,0] = torch.tensor(r_squared_mean_along_batches[0])
        test_metrics[model_idx,test_idx,1] = torch.tensor(r_squared_var_along_batches[0])
        test_metrics[model_idx,test_idx,2] = torch.tensor(rmse_mean_along_batches[0])
        test_metrics[model_idx,test_idx,3] = torch.tensor(rmse_var_along_batches[0])

        upper_r_squared = np.ones(num_test_robots)* ( r_squared_mean_along_batches + r_squared_var_along_batches )
        lower_r_squared = np.ones(num_test_robots)* ( r_squared_mean_along_batches - r_squared_var_along_batches)
        upper_rmse = np.ones(num_test_robots)* ( rmse_mean_along_batches + rmse_var_along_batches )
        lower_rmse = np.ones(num_test_robots)* ( rmse_mean_along_batches - rmse_var_along_batches)

        # -------------------------------Metrics PLOT -----------------------------

        log.write('\n\n' + suptitle1 + '\n' + suptitle2 + '\n')
        log.write('\nRMSE_mean ' + str(round(rmse_mean_along_batches[0],6)) + '\nRMSE_var ' + str(round(rmse_var_along_batches[0],6)))
        log.write('\nR2_mean ' + str(round(r_squared_mean_along_batches[0],4)) + '\nR2_var ' + str(round(r_squared_var_along_batches[0],3))+ '\n')

        minR_squared_idx = np.argmin(r_squared_mean_along_y_dimensions)
        maxR_squared_idx = np.argmax(r_squared_mean_along_y_dimensions)

        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,15))
        fig.suptitle('$R^{2}$ & $RMSE$ index along robots'+ '\n MODEL: '
                    + model_name + '\n tested on: ' + suptitle)

        ax2.plot(rmse_mean_along_y_dimensions,linewidth=1.2)
        ax2.plot(rmse_mean_along_batches,'r',linewidth=2,linestyle='dotted')
        ax2.fill_between( range(0,num_test_robots), upper_rmse, lower_rmse, facecolor='blue', alpha=0.25)
        ax2.grid(True)

        ax1.plot(r_squared_mean_along_y_dimensions,linewidth=1.2)
        ax1.plot(r_squared_mean_along_batches,'r',linewidth=2,linestyle='dotted') 
        ax1.scatter(minR_squared_idx,r_squared_mean_along_y_dimensions[minR_squared_idx],marker='o', color='red', s=30)
        ax1.scatter(maxR_squared_idx,r_squared_mean_along_y_dimensions[maxR_squared_idx],marker='o', color='green', s=30)
        ax1.fill_between( range(0,num_test_robots), upper_r_squared, lower_r_squared, facecolor='blue', alpha=0.25)
        ax1.grid(True)

        ax1.set_title('$R^{2}$')
        ax1.set_ylim([r_squared_mean_along_y_dimensions[minR_squared_idx], 1.2])
        ax2.set_ylim([0, .1])
        ax2.text(int(num_test_robots/3), 0.05, 'RMSE:' +str(round(rmse_mean_along_batches[0],6)), fontsize = 18, bbox = dict(facecolor = 'red', alpha = 0.8) ) 
        ax1.text(int(num_test_robots/3), 1.05, '$R^{2}$ :' +str(round(r_squared_mean_along_batches[0],4)), fontsize = 18,bbox = dict(facecolor = 'red', alpha = 0.8)) 

        ax1.set(xlabel='$i_{th}$ robot ')

        ax2.set_title('$RMSE$')
        ax2.set(xlabel='$i_{th}$ robot')

        if save_figure == True:

            model_path = model_name.replace(".pt"," ")
            model_path = Path(fig_path / f"{model_path}")
            model_path.mkdir(exist_ok=True)
            test_path = list_of_available_tensors[test_idx].replace(".pt"," ")
            test_path = Path(model_path / f"{test_path}")
            test_path.mkdir(exist_ok=True)

            plt.savefig(test_path / f"R2_rmse.png")
            plt.close()

        # -------------------------------Prediction PLOT ----------------------------- 
            
        test_idxs = [minR_squared_idx, maxR_squared_idx]
        label_idxs =['Worst_R2','Best_R2']    

        for idx in range(len(test_idxs)):
            
            fig, axs = plt.subplots(int(ny/2), 2,figsize=(20, 20))
            font = {'family' : 'DejaVu Sans',
                    'weight' : 'normal',
                    'size'   : 15}
            plt.rc('font', **font)
            
            fig.suptitle('Prediction of robot number'+ str(test_idxs[idx]) + ' ['+ str(label_idxs[idx])+']' + '\n MODEL: '+ model_name + '\n tested on: ' + suptitle ,size = 20)

            labels_coordinates = ["$x$","$y$","$z$" ,"$Q_1$","$Q_2$","$Q_3$" ,"$Q_4$","q0","q1","q2","q3","q4","q5","q6"]
            labels_pred = ["$\hat x$","$\hat y$","$\hat z$","$\hat Q_1$","$ \hat Q_2$","$\hat Q_3$" ,"$ \hat Q_4$"
                        ,"$ \hat q0$","$ \hat q1$","$ \hat q2$","$ \hat q3$","$ \hat q4$","$ \hat q5$","$ \hat q6$"]
            labels_error = ["$x - \hat x$","$y - \hat y$","$z - \hat z$","$Q_1 - \hat Q_1$","$ Q_2 - \hat Q_2$",
                            "$Q_3 - \hat Q_3$" ,"$ Q_4 -  \hat Q_4$","$ q0 -  \hat q0$","$ q1 -  \hat q1$","$ q2 -  \hat q2$"
                            ,"$ q3 -  \hat q3$","$ q4 -  \hat q4$","$ q5 -  \hat q5$","$ q6 -  \hat q6$"]
            k = 0
            idx_plot = test_idxs[idx]

            t1 = np.arange(1, cfg.seq_len_ctx) 

            for i in range(int(ny/2)):
                for j in range(int(2)):
                    
                    axs[i,j].plot(t1, position[idx,0:cfg.seq_len_ctx-1,k] ,'k',linewidth=2)

                    axs[i,j].fill_between( t1, min(position[idx, :, k]), max(position[idx, :, k]), facecolor='lime', alpha=0.2)

                    axs[i,j].plot(t, y_ground_truth[idx, :, k] ,'k', label=labels_coordinates[k],linewidth=2)
                    axs[i,j].plot(t, y_predicted[idx, :, k], 'b', label=labels_pred[k],linewidth=2)
                    axs[i,j].plot(t, sim_error[idx, :, k], 'r', label=labels_error[k])

                    axs[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    axs[i,j].grid(True)
                    k=k+1


            plt.xlabel("time step (-)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.tight_layout()
            

            if save_figure == True:

                plt.savefig(test_path / f"{label_idxs[idx]}.png")

                plt.close()
        
        # -------------------------------Prediction Context ----------------------------- 
                
        t_context = np.arange(1, cfg.seq_len_ctx) 
        t_prediction = np.arange(1, single_u_new.shape[0]+1) + cfg.seq_len_ctx
        t_total = np.arange(0, position.shape[1])

        idx = int(np.ceil(torch.rand(1).uniform_(0,int(len(rmse_mean_along_y_dimensions)-1)).numpy()))
        
        fig, axs = plt.subplots(int(ny/2), 2,figsize=(20, 20))
        fig.suptitle('Prediction of y - batch idx '+ str(idx),size = 20,y = .99)

        k = 0

        for i in range(int(ny/2)):
            for j in range(int(2)):

                # Plot context - green
                # axs[i,j].plot(t_context, position[idx,0:cfg.seq_len_ctx-1,k] ,'k',linewidth=2)
                min_value = min (min(position[idx, :, k]),min(sim_error[idx, :, k]))
                max_value = max (max(position[idx, :, k]),max(sim_error[idx, :, k]))
                axs[i,j].axvline(x = t_prediction[-1], color = 'k', linestyle='--')
                axs[i,j].axvline(x = t_context[-1], color = 'k', linestyle='--')

                axs[i,j].fill_between( t_context, min_value, max_value, facecolor='lime', alpha=0.2)

                # Plot prediction & error
                axs[i,j].plot(t_prediction, y_predicted[idx, :, k], 'b', label=labels_pred[k],linewidth=2)
                axs[i,j].plot(t_prediction, sim_error[idx, :, k], 'r', label=labels_error[k])

                # Plot ground truth 
                # axs[i,j].plot(t_prediction, y_ground_truth[idx, :, k] ,'k', label=labels_coordinates[k],linewidth=2)

                #Plot entire Signal
                axs[i,j].plot(t_total, position[idx, :, k], 'k', label=labels_error[k])

                axs[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                axs[i,j].grid(True)
                k=k+1

        plt.xlabel("time step (-)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()

        if save_figure == True:
            plt.savefig(test_path / f"robot_{idx}_ctx.png")

# -----------------------------------------------------------------------------------
                
log.write("\n\n Based on R^2 mean:\n\n ")
print("\n Based on R^2 mean:\n\n ")
for test_idx in range(len(list_of_available_tensors)):
    print("\nFor " + list_of_available_tensors[test_idx])
    log.write("\nFor " + list_of_available_tensors[test_idx])
    max_r2_idx = torch.argmax(test_metrics[:,test_idx,0]).item()
    print(" The best model is " + list_of_available_models[max_r2_idx]+ str(test_metrics[max_r2_idx,test_idx,0]))
    log.write("The best model is " + list_of_available_models[max_r2_idx] + str(test_metrics[max_r2_idx,test_idx,0]))

log.write("\n\n Based on R^2 var:\n\n ")
print("\n\nBased on R^2 var:\n\n ")
for test_idx in range(len(list_of_available_tensors)):
    print("\nFor " + list_of_available_tensors[test_idx])
    log.write("\nFor " + list_of_available_tensors[test_idx])
    min_r2_var_idx = torch.argmin(test_metrics[:,test_idx,1]).item()
    print(" The best model is " + list_of_available_models[min_r2_var_idx] + str(test_metrics[min_r2_var_idx,test_idx,1]))
    log.write("The best model is " + list_of_available_models[min_r2_var_idx] +str(test_metrics[min_r2_var_idx,test_idx,1] ))


log.write("\n\n Based on RMSE mean:\n\n ")
print("\n Based on RMSE mean:\n\n ")
for test_idx in range(len(list_of_available_tensors)):
    print("\nFor " + list_of_available_tensors[test_idx])
    log.write("\nFor " + list_of_available_tensors[test_idx])
    min_rmse_idx = torch.argmin(test_metrics[:,test_idx,2]).item()
    print(" The best model is " + list_of_available_models[min_rmse_idx] + str(test_metrics[min_rmse_idx,test_idx,2]))
    log.write("The best model is " + list_of_available_models[min_rmse_idx] + str(test_metrics[min_rmse_idx,test_idx,2]))

log.write("\n\n Based on RMSE var:\n\n ")
print("\n Based on RMSE var:\n\n ")
for test_idx in range(len(list_of_available_tensors)):
    print("\nFor " + list_of_available_tensors[test_idx])
    log.write("\nFor " + list_of_available_tensors[test_idx])
    min_rmse_var_idx = torch.argmin(test_metrics[:,test_idx,3]).item()
    print(" The best model is " + list_of_available_models[min_rmse_var_idx] + str(test_metrics[min_rmse_var_idx,test_idx,3]))
    log.write("The best model is " + list_of_available_models[min_rmse_var_idx] + str(test_metrics[min_rmse_var_idx,test_idx,3]))
log.close()


# ------------------------------------

x = np.arange(0,len(list_of_available_tensors))

for i in range(len(list_of_available_models)):
    fig, axs = plt.subplots(1,1,figsize=(18,12))
    
    plt.suptitle('Model '+ str(list_of_available_models[i]) +' along different tests',fontsize=20)
    plt.errorbar(x, test_metrics[i,:,0],fmt = 'o', yerr=test_metrics[i,:,1], ecolor= 'red', elinewidth = 2, capsize=10)
    plt.plot(torch.zeros(x.shape),'k--')
    plt.plot(torch.ones(x.shape),'b--')

    tickvalues = range(0,len(list_of_available_tensors)) 
    list_labels = []

    for k in range(len(list_of_available_tensors)): 
        list_labels.append( "test_" + str(k))
    
    plt.xticks(ticks = tickvalues, labels = list_labels ,fontsize=18, rotation = -10)
    plt.yticks(fontsize=20)
    plt.ylabel('$R^{2}$',size=20,rotation=0)
    axs.set_ylim([-1.2, 1.5])
    plt.grid(True,alpha=0.5)
    plt.subplots_adjust(top=0.93,bottom=0.179,left=0.139,right=0.867,hspace=0.2,wspace=0.2)


    # For all the tested models, print a recap of 
    for j in range(len(x)):

        training_frequency = 0.15 #extract from the file !!! WIP 
        training_mass_bounds = (10,10)

        current_test = list_of_available_tensors[j]
        frequency_index = current_test.find('f_')
        randomization_index = current_test.find('mass_') 
    
        # --------------------------------------------------------------------------
        try:
            test_frequency = current_test[frequency_index+2:frequency_index+6]
            # if test_frequency is 0.1 --> 0_1_ would become 01, so remove _ at the end
            if test_frequency[-1] == '_': 
                test_frequency = test_frequency[:len(test_frequency)-1]

            test_frequency=test_frequency.replace('_','.')
            test_frequency = float(test_frequency)
            if test_frequency <= training_frequency:
                suptitle1 = '\nID frequency'
                flag1 = True
            else:
                suptitle1 = '\n f: ' + current_test[frequency_index+2:frequency_index+5].replace('_','.')+' Hz'
                flag1 = False
            
            test_bounds = current_test[randomization_index+5:randomization_index+10]
            test_bounds = test_bounds.replace('p','').replace('.','')
            test_bounds = list(test_bounds.split("_"))
            test_lower_bound = float(test_bounds[0])
            test_higher_bound = float(test_bounds[1])

            # if current_test[randomization_index+5:randomization_index+10].replace('p','') == mass_bounds:

            if test_lower_bound <= training_mass_bounds[0] and test_higher_bound <= training_mass_bounds[1] :
                suptitle2 = '\nID randomization'
                flag2 = True
            else:
                suptitle2 = '\n mass: +' +current_test[randomization_index+5:randomization_index+10].replace('_','/-').replace('.p','') +' % '
                flag2 = False

        except ValueError:
                suptitle1 = '\nf: Task XY'
                flag1 = False
                flag2 = False
                suptitle1 = '\nf: - '
        # ----------------------------------------------------------------------------

        suptitle = suptitle1 + suptitle2 
        flag = flag1 and flag2

        if flag:
            bbox_dict = dict(facecolor = 'green', alpha = 0.4)
        else:
            bbox_dict =dict(facecolor = 'red', alpha = 0.7)

        plt.text(j-0.2, -0.7, '$R^2$: ' +str(round(test_metrics[i,j,0].item(),4))+'\n'+'$\sigma_{R^{2}}$: ' +
                 str(round(test_metrics[i,j,1].item(),4))+suptitle, fontsize = 22, bbox = bbox_dict  ) 
        plt.xlim(-.25, j +  .25)

    if save_figure == True:
        output_name = f"benchmark_{str(list_of_available_models[i]).replace('.pt','')}.png"
        plt.tight_layout()
        plt.savefig(fig_path / output_name )
    else:
        plt.show()

torch.save(test_metrics,'./fig/metrics_models.pt')

           
