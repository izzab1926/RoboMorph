from pathlib import Path
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformer_sim import Config, TSTransformer

from extrapolation import check_ID_OOD
#import tqdm
import metrics
from torch.utils.data import random_split
import os   
import argparse

parser = argparse.ArgumentParser(description='Meta system identification with transformers - IsaacGym')

# Overall
parser.add_argument('--test-folder', type=str, default="test", metavar='test',
                    help='test-folder')
parser.add_argument('--figure-folder', type=str, default="recap_mul_horizon", metavar='figure',
                    help='Figure output')
parser.add_argument('--model-folder', type=str, default="out_ds2", metavar='model',
                    help='Figure output')

parser.add_argument('--save-figure', action='store_true', default=True,
                    help='Save vigure in png')
parser.add_argument('--plot-predictions-and-metrics', action='store_true', default=True,
                    help='Figure output')

parser.add_argument('--num-test', type=int, default=1000, metavar='num',
                    help='number of wanted test trajectories for each file')

test_cfg,unparsed = parser.parse_known_args() 

# -----------------------------------------------
# test_cfg.figure_folder 
# test_cfg.test_folder 
# test_cfg.model_folder 

fig_path = Path(f"{test_cfg.figure_folder}")
fig_path.mkdir(exist_ok=True)

log = open(fig_path / "log.txt","w")

torch.manual_seed(420)
np.random.seed(430)
torch.manual_seed(420)
np.random.seed(430)

# Overall settings
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

torch.set_float32_matmul_precision("high") 
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# ---------------------------- Testing file ----------------------------

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

# --------------------------- Testing model --------------------------------------
model_path = os.path.abspath(f"./{test_cfg.model_folder}")
list_of_available_models = os.listdir(model_path)
print("\nThese are the available models in "+str(model_path) +":\n")
log.write("\nThese are the available models in "+str(model_path) +":\n")
for i in range(len(list_of_available_models)):
    single_model_file_path = os.path.join(model_path, list_of_available_models[i]) 
    print("--> ",list_of_available_models[i])
    log.write("--> " + list_of_available_models[i]+ "\n")
# ---------------------------------------------------------------------------------
    
for model_idx in range(len(list_of_available_models)):

    model_name = list_of_available_models[model_idx]
    print(" \n --> Testing model: ", model_name )
    log.write("\n\n *********** Testing model: "+ model_name + ' ************ \n')
    
    # load model's wieghts (partition_**.pt file)
    out_dir = Path(out_dir) # "out"
    exp_data = torch.load(out_dir/model_name, map_location="cpu")
    cfg = exp_data["cfg"]
    # It is possible to test on different scales, even it was trained on a different partition 
    # cfg.seq_len_ctx = 500
    # cfg.seq_len_new = 150

    # VARIABLE 
    cfg.seq_len_ctx = 200

    start_prediction_horizion = 200
    step_prediction = 50

    total_time = 1000 
    max_horizon_length = total_time - cfg.seq_len_ctx

    end_horizon = max_horizon_length

    end_horizon +=1 # (for approximation)
    total_time +=1
    prediction_horizons = np.arange(start_prediction_horizion,end_horizon,step_prediction)

    # test metrics of each model
    test_metrics = torch.zeros((len(prediction_horizons),len(list_of_available_tensors),8))
    list_of_RMSE_total = []
    list_of_R2_total = []

    for idx_len_prediction in range(len(prediction_horizons)):
        
        # test all the available files 
        # Empty list for boxplot
        list_of_R2 = []
        list_of_RMSE = []

        print(" \n --> Testing prediction horizon: ", prediction_horizons[idx_len_prediction] )
        log.write("\n --> Testing prediction horizon: " + str(prediction_horizons[idx_len_prediction]))

        cfg.seq_len_new = prediction_horizons[idx_len_prediction]
        seq_len = cfg.seq_len_ctx + cfg.seq_len_new


        model_args = exp_data["model_args"]
        conf = Config(**model_args)
        model = TSTransformer(conf).to(device)
        model.load_state_dict(exp_data["model"]);

        # test all the available files 

        for test_idx in range(len(list_of_available_tensors)): 

            single_pt_file_path = os.path.join(tensors_path, list_of_available_tensors[test_idx]) 

            log.write("\nTEST DATASET: " + list_of_available_tensors[test_idx])
            loaded = torch.load(single_pt_file_path,map_location=device) #

            # Import All the 1000 steps, eventhough prediction may be on a subset
            @ torch.no_grad()
            def loading():
                if loaded['control_action'].shape[0] == 1001:
                    control_action_extracted = loaded['control_action'][1:,:test_cfg.num_test,:7] #time x num_envs x dofs
                else:
                    control_action_extracted = loaded['control_action'][:,:test_cfg.num_test,:7]
                position_extracted = loaded['position']
                position = torch.movedim(position_extracted.to('cpu'),-2,-3)
                control_action = torch.movedim(control_action_extracted.to('cpu'),-2,-3)
                return control_action,position
            
            control_action,position = loading()
            num_test = control_action.shape[0] 
            ny = position.shape[2]

            if test_idx == 0 and model_idx ==0 and idx_len_prediction ==0:
                #random_idx for everyone!
                random_idx = int(np.ceil(torch.rand(1).uniform_(0,num_test-1).numpy()))
                print("Number of tests: ", num_test, "random idx: ", random_idx)

            # Initializing folders
            model_path = model_name.replace(".pt"," ")
            model_path = Path(fig_path / f"{model_path}")
            model_path.mkdir(exist_ok=True)
            test_path = list_of_available_tensors[test_idx].replace(".pt",f"_horizon{str(cfg.seq_len_new)}")
            test_path = Path(model_path / f"{test_path}")
            test_path.mkdir(exist_ok=True)
        
            #------------------------- initializing errors ---------------------------------

            y_ground_truth = torch.empty(position[:num_test,cfg.seq_len_ctx:seq_len,:].shape,device=device)
            y_predicted = torch.empty(position[:num_test,cfg.seq_len_ctx:seq_len,:].shape,device=device)
            sim_error = torch.empty(position[:num_test,cfg.seq_len_ctx:seq_len,:].shape,device=device)

            # print(" \n------------- Calculating errors... ----------------- ")

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

            # -------------------------------Metrics calculation -----------------------------
            
            t = np.arange(1, single_u_new.shape[0]+1) + cfg.seq_len_ctx
            y_ground_truth = y_ground_truth.to("cpu").detach().numpy()
            y_predicted = y_predicted.to("cpu").detach().numpy()
            sim_error = sim_error.to("cpu").detach().numpy()
            
            RMSE = metrics.rmse(y_ground_truth, y_predicted, time_axis=1)
            NRMSE = metrics.nrmse(y_ground_truth, y_predicted, time_axis=1)
            fit_index = metrics.fit_index(y_ground_truth, y_predicted, time_axis=1)
            r_squared = metrics.r_squared(y_ground_truth, y_predicted, time_axis=1)

            #R2
            r_squared_mean_along_y_dimensions = torch.empty((num_test,1),device = device)
            r_squared_std_along_y_dimensions = torch.empty((num_test,1),device = device)
            #RMSE
            RMSE_mean_along_y_dimensions = torch.empty((num_test,1),device = device)
            RMSE_std_along_y_dimensions = torch.empty((num_test,1),device = device)
            # NRMSE
            NRMSE_mean_along_y_dimensions = torch.empty((num_test,1),device = device)
            NRMSE_std_along_y_dimensions = torch.empty((num_test,1),device = device)
            # fit index
            fit_index_mean_along_y_dimensions = torch.empty((num_test,1),device = device)
            fit_index_std_along_y_dimensions = torch.empty((num_test,1),device = device)
            
            for i in range(r_squared.shape[0]):
                r_squared_mean_along_y_dimensions[i]= torch.tensor(r_squared[i,:].mean())
                r_squared_std_along_y_dimensions[i] = torch.tensor(r_squared[i,:].std())

                RMSE_mean_along_y_dimensions[i]= torch.tensor(RMSE[i,:].mean())
                RMSE_std_along_y_dimensions[i] = torch.tensor(RMSE[i,:].std())

                NRMSE_mean_along_y_dimensions[i]= torch.tensor(NRMSE[i,:].mean())
                NRMSE_std_along_y_dimensions[i] = torch.tensor(NRMSE[i,:].std())

                fit_index_mean_along_y_dimensions[i]= torch.tensor(fit_index[i,:].mean())
                fit_index_std_along_y_dimensions[i] = torch.tensor(fit_index[i,:].std())
            #R2
            r_squared_mean_along_y_dimensions = r_squared_mean_along_y_dimensions.to("cpu").detach().numpy()
            r_squared_std_along_y_dimensions = r_squared_std_along_y_dimensions.to("cpu").detach().numpy()
            #RMSE
            RMSE_mean_along_y_dimensions = RMSE_mean_along_y_dimensions.to("cpu").detach().numpy()
            RMSE_std_along_y_dimensions = RMSE_std_along_y_dimensions.to("cpu").detach().numpy()
            #NMSE
            NRMSE_mean_along_y_dimensions = NRMSE_mean_along_y_dimensions.to("cpu").detach().numpy()
            NRMSE_std_along_y_dimensions = NRMSE_std_along_y_dimensions.to("cpu").detach().numpy()
            #Fit index
            fit_index_mean_along_y_dimensions = fit_index_mean_along_y_dimensions.to("cpu").detach().numpy()
            fit_index_std_along_y_dimensions = fit_index_std_along_y_dimensions.to("cpu").detach().numpy()

            # calculating vectors for plotting
            r_squared_mean_along_batches = np.ones(num_test)* r_squared_mean_along_y_dimensions.mean()
            r_squared_std_along_batches = np.ones(num_test)* r_squared_std_along_y_dimensions.mean()
            upper_r2 = np.ones(num_test)* ( r_squared_mean_along_batches + r_squared_std_along_batches )
            lower_r2 = np.ones(num_test)* ( r_squared_mean_along_batches - r_squared_std_along_batches)

            RMSE_mean_along_batches = np.ones(num_test)* RMSE_mean_along_y_dimensions.mean()
            RMSE_std_along_batches = np.ones(num_test)* RMSE_std_along_y_dimensions.mean()
            upper_RMSE = np.ones(num_test)* ( RMSE_mean_along_batches + RMSE_std_along_batches )
            lower_RMSE = np.ones(num_test)* ( RMSE_mean_along_batches - RMSE_std_along_batches)

            fit_index_mean_along_batches = np.ones(num_test)*fit_index_mean_along_y_dimensions.mean()
            fit_index_std_along_batches = np.ones(num_test)*fit_index_std_along_y_dimensions.mean()
            upper_fit = np.ones(num_test)* ( fit_index_mean_along_batches + fit_index_std_along_batches )
            lower_fit = np.ones(num_test)* ( fit_index_mean_along_batches - fit_index_std_along_batches)

            NRMSE_mean_along_batches = np.ones(num_test)*NRMSE_mean_along_y_dimensions.mean()
            NRMSE_std_along_batches = np.ones(num_test)*NRMSE_std_along_y_dimensions.mean()
            upper_NRMSE = np.ones(num_test)* ( NRMSE_mean_along_batches + NRMSE_std_along_batches)
            lower_NRMSE = np.ones(num_test)* ( NRMSE_mean_along_batches - NRMSE_std_along_batches)
            
            # highest is better
            worst_R2_idx = np.argmin(r_squared_mean_along_y_dimensions)
            best_R2_idx = np.argmax(r_squared_mean_along_y_dimensions)
            worst_fit_index_idx = np.argmin(fit_index_mean_along_y_dimensions)
            best_fit_index_idx = np.argmax(fit_index_mean_along_y_dimensions)
            # Lowest is better
            worst_RMSE_idx = np.argmax(RMSE_mean_along_y_dimensions)
            best_RMSE_idx = np.argmin(RMSE_mean_along_y_dimensions)
            worst_NRMSE_idx = np.argmax(NRMSE_mean_along_y_dimensions)
            best_NRMSE_idx = np.argmin(NRMSE_mean_along_y_dimensions)

            # WIP 
            test_metrics[idx_len_prediction,test_idx,0] = torch.tensor(r_squared_mean_along_y_dimensions.mean())
            test_metrics[idx_len_prediction,test_idx,1] = torch.tensor(r_squared_std_along_y_dimensions.mean())
            test_metrics[idx_len_prediction,test_idx,2] = torch.tensor(RMSE_mean_along_y_dimensions.mean())
            test_metrics[idx_len_prediction,test_idx,3] = torch.tensor(RMSE_std_along_y_dimensions.mean())
            test_metrics[idx_len_prediction,test_idx,4] = torch.tensor(NRMSE_mean_along_y_dimensions.mean())
            test_metrics[idx_len_prediction,test_idx,5] = torch.tensor(NRMSE_std_along_y_dimensions.mean())
            test_metrics[idx_len_prediction,test_idx,6] = torch.tensor(fit_index_mean_along_y_dimensions.mean())
            test_metrics[idx_len_prediction,test_idx,7] = torch.tensor(fit_index_std_along_batches.mean())

            # -------------------------------Metrics PLOT -----------------------------

            if plot_predictions_and_metrics:
            
                fig,[[ax1, ax2],[ax3, ax4]]  = plt.subplots(2, 2,figsize=(15, 15), dpi=300)
                fig.suptitle('Metrics'+ '\n MODEL: '+ model_name + 
                            '\n tested on: ' + list_of_available_tensors[test_idx] ,size = 22)

                ax2.plot(RMSE_mean_along_y_dimensions,linewidth=1.5)
                ax2.plot(RMSE_mean_along_batches,'r',linewidth=1.5,linestyle='dotted') #{labels_coordinates[k]}
                ax2.fill_between( range(0,control_action.shape[0]), upper_RMSE, lower_RMSE, facecolor='yellow', alpha=0.2)
                ax1.scatter(worst_R2_idx,r_squared_mean_along_y_dimensions[worst_R2_idx],marker='o', color='red', s=30)
                ax1.scatter(best_R2_idx,r_squared_mean_along_y_dimensions[best_R2_idx],marker='o', color='green', s=30)   

                ax1.plot(r_squared_mean_along_y_dimensions,linewidth=1.5)
                ax1.plot(r_squared_mean_along_batches,'r',linewidth=1.5,linestyle='dotted') #{labels_coordinates[k]}
                ax1.scatter(worst_R2_idx,r_squared_mean_along_y_dimensions[worst_R2_idx],marker='o', color='red', s=30)
                ax1.scatter(best_R2_idx,r_squared_mean_along_y_dimensions[best_R2_idx],marker='o', color='green', s=30)   
                ax1.fill_between( range(0,control_action.shape[0]), upper_r2, lower_r2, facecolor='yellow', alpha=0.2)
                ax2.grid(True)
                ax1.grid(True)
                ax1.set_title('$R^{2}$')
                ax2.set_title('$RMSE$')
                ax1.set(xlabel='$i_{th}$ robot ')
                ax2.set(xlabel='$i_{th}$ robot')

                dimension_font = 15
                bbox_dict = dict(facecolor = 'yellow', alpha = 1)

                #R2
                mid = (r_squared_mean_along_y_dimensions[worst_R2_idx] + 
                    r_squared_mean_along_batches[0])/2
                if -1.1 < r_squared_mean_along_batches[0] < 1.1 and r_squared_mean_along_y_dimensions[worst_R2_idx] < -1.1:
                    mid = 0
                ax1.text((num_test/3), mid, 
                        "$R^2_{mean}$: " +
                        str(round(r_squared_mean_along_batches[0].item(),4)), 
                        fontsize = dimension_font, bbox = bbox_dict ) 
                #RMSE
                mid = (RMSE_mean_along_y_dimensions[worst_RMSE_idx] + 
                    RMSE_mean_along_batches[0])/2
                ax2.text((num_test/3),mid,
                        "$RMSE_{mean}$: " +
                        str(round(RMSE_mean_along_batches[0].item(),4)), 
                        fontsize = dimension_font, bbox = bbox_dict) 

                #fit index
                mid = (fit_index_mean_along_y_dimensions[worst_fit_index_idx] + 
                    fit_index_mean_along_batches[0])/2
                ax3.text((num_test/3), mid, 
                        "$fit\,\,index_{mean}$: " +
                        str(round(fit_index_mean_along_batches[0].item(),4)), 
                        fontsize = dimension_font, bbox = bbox_dict  ) 
                
                #NRMSE
                mid = (NRMSE_mean_along_y_dimensions[worst_NRMSE_idx] + 
                    NRMSE_mean_along_batches[0])/2
                ax4.text((num_test/3), mid, 
                        "$NRMSE_{mean}$: " +
                        str(round(NRMSE_mean_along_batches[0].item(),4)),
                        fontsize = dimension_font, bbox = bbox_dict  ) 

                # Set limits 
                if -1.1 < r_squared_mean_along_batches[0] < 1.1:
                    ax1.set_ylim([-1.1, r_squared_mean_along_y_dimensions[best_R2_idx]*1.2])

                if 0 < fit_index_mean_along_batches[0] < 100:
                    ax3.set_ylim([0, fit_index_mean_along_y_dimensions[best_fit_index_idx]*1.1])

                ax3.plot(fit_index_mean_along_y_dimensions,linewidth=1.5)
                ax3.plot(fit_index_mean_along_batches,'r',linewidth=1.5,linestyle='dotted') #{labels_coordinates[k]}
                ax3.scatter(worst_fit_index_idx,fit_index_mean_along_y_dimensions[worst_fit_index_idx],marker='o', color='red', s=30)
                ax3.scatter(best_fit_index_idx,fit_index_mean_along_y_dimensions[best_fit_index_idx],marker='o', color='green', s=30)
                ax3.fill_between( range(0,num_test), upper_fit, lower_fit, facecolor='yellow', alpha=0.2)
                ax4.plot(NRMSE_mean_along_y_dimensions,linewidth=1.5)
                ax4.plot(NRMSE_mean_along_batches,'r',linewidth=1.5,linestyle='dotted') #{labels_coordinates[k]}
                ax4.fill_between( range(0,num_test), upper_NRMSE, lower_NRMSE, facecolor='yellow', alpha=0.2)
                ax3.grid(True)
                ax4.grid(True)
                ax3.set_title('$fit\,\,index\,\%$')
                ax4.set_title('$NRMSE$')
                ax4.set(xlabel='$i_{th}$ robot')
                ax4.set(xlabel='$i_{th}$ robot')

                fig.tight_layout(pad = 1.2 )

                if save_figure == True:

                    model_path = model_name.replace(".pt"," ")
                    model_path = Path(fig_path / f"{model_path}")
                    model_path.mkdir(exist_ok=True)
                    test_path = list_of_available_tensors[test_idx].replace(".pt",f"_horizon{str(cfg.seq_len_new)}")
                    test_path = Path(model_path / f"{test_path}")
                    test_path.mkdir(exist_ok=True)
                    plt.savefig(test_path / f"metrics.png")
                    plt.close()

                # -------------------------------Prediction PLOT ----------------------------- 
                    
                labels_coordinates = ["$x$","$y$","$z$" ,"$X$","$Y$","$Z$" ,"$W$","q0","q1","q2","q3","q4","q5","q6"]
                labels_pred = ["$\hat x$","$\hat y$","$\hat z$","$\hat X$","$ \hat Y$","$\hat Z$" ,"$ \hat W$"
                            ,"$ \hat q0$","$ \hat q1$","$ \hat q2$","$ \hat q3$","$ \hat q4$","$ \hat q5$","$ \hat q6$"]
                labels_error = ["$x - \hat x$","$y - \hat y$","$z - \hat z$","$X - \hat X$","$ Y - \hat Y$",
                                "$Z - \hat Z$" ,"$ W -  \hat W$","$ q0 -  \hat q0$","$ q1 -  \hat q1$","$ q2 -  \hat q2$"
                                ,"$ q3 -  \hat q3$","$ q4 -  \hat q4$","$ q5 -  \hat q5$","$ q6 -  \hat q6$"]
   

                test_idxs = [worst_RMSE_idx, best_RMSE_idx,worst_R2_idx, best_R2_idx,random_idx]
                label_idxs =['Worst_RMSE','Best_RMSE','Worst_R2','Best_R2','random']  

                for idx in range(len(test_idxs)):

                    fig, axs = plt.subplots(int(ny/2), 2,figsize=(20, 20))
                    fig.suptitle('Prediction of y - robot idx '+ str(test_idxs[idx]) + 
                                ' ['+ str(label_idxs[idx])+']',size = 24,y = .99)
                    k = 0

                    idx_plot = test_idxs[idx]

                    for j in range(int(2)):
                        for i in range(int(ny/2)):
                            axs[i,j].plot(t, y_ground_truth[idx_plot, :, k] ,'k', label=labels_coordinates[k],linewidth=2)
                            axs[i,j].plot(t, y_predicted[idx_plot, :, k], 'b', label=labels_pred[k],linewidth=2)
                            axs[i,j].plot(t, sim_error[idx_plot, :, k], 'r', label=labels_error[k])
                            axs[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                            axs[i,j].grid(True)
                            k=k+1

                    plt.xlabel("time step (-)")
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    plt.tight_layout( pad = 1.2 )

                    if save_figure == True:

                        plt.savefig(test_path / f"{label_idxs[idx]}.png")

                        plt.close()
            
            # -------------------------------Prediction Context ----------------------------- 

                t_context = np.arange(1, cfg.seq_len_ctx) 
                t_prediction = np.arange(1, single_u_new.shape[0]+1) + cfg.seq_len_ctx
                t_total = np.arange(0, position.shape[1])

                fig, axs = plt.subplots(int(ny/2), 2,figsize=(20, 20))
                fig.suptitle('Prediction of y - robot idx '+ str(random_idx),size = 24,y = .99)

                k = 0
                
                for j in range(int(2)):
                    for i in range(int(ny/2)):
                        # Plot context - green
                        # axs[i,j].plot(t_context, position[idx,0:cfg.seq_len_ctx-1,k] ,'k',linewidth=2)
                        min_value = min (min(position[random_idx, :, k]),min(sim_error[random_idx, :, k]))
                        max_value = max (max(position[random_idx, :, k]),max(sim_error[random_idx, :, k]))
                        axs[i,j].axvline(x = t_prediction[-1], color = 'k', linestyle='--')
                        axs[i,j].axvline(x = t_context[-1], color = 'k', linestyle='--')
                        axs[i,j].fill_between( t_context, min_value, max_value, facecolor='lime', alpha=0.2)
                
                        # Plot prediction & error
                        axs[i,j].plot(t_prediction, y_predicted[random_idx, :, k], 'b', label=labels_pred[k],linewidth=2)
                        axs[i,j].plot(t_prediction, sim_error[random_idx, :, k], 'r', label=labels_error[k])
                        # Plot entire Signal
                        axs[i,j].plot(t_total, position[random_idx, :, k], 'k', label=labels_coordinates[k])

                        axs[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                        axs[i,j].grid(True)

                        k=k+1
                            

                plt.xlabel("time step (-)")
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.tight_layout( pad = 1.2)

                if save_figure == True:
                    plt.savefig(test_path / f"robot_{random_idx}_ctx.png")
                    plt.close()

        # ----------------------- zoom of first coordinate -------------------------------
                font = {'family' : 'DejaVu Sans',
                        'weight' : 'normal',
                        'size'   : 22}
                plt.rc('font', **font)

                fig2, axs2 = plt.subplots(1, 1,figsize=(18, 4))
                fig2.suptitle('Prediction of x coordinate - Prediction horizon: ' +str(prediction_horizons[idx_len_prediction]) +' steps',size = 24,y = 0.95)
                
                k = 0
                min_value = min (min(position[random_idx, :, k]),min(sim_error[random_idx, :, k]))
                max_value = max (max(position[random_idx, :, k]),max(sim_error[random_idx, :, k]))
                axs2.axvline(x = t_prediction[-1], color = 'k', linestyle='--')
                axs2.axvline(x = t_context[-1], color = 'k', linestyle='--')
                axs2.fill_between( t_context, min_value, max_value, facecolor='lime', alpha=0.2)
                # Plot prediction & error
                axs2.plot(t_prediction, y_predicted[random_idx, :, k], 'b', label=labels_pred[k],linewidth=2)
                axs2.plot(t_prediction, sim_error[random_idx, :, k], 'r', label=labels_error[k])
                # Plot entire Signal
                axs2.plot(t_total, position[random_idx, :, k], 'k', label=labels_coordinates[k])
                axs2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                axs2.grid(True)
                plt.xlabel("time step (-)")
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.tight_layout( pad = 1.2)
                if save_figure == True:
                    plt.savefig(model_path / f"robot_{random_idx}_ctx_ZOOM_pred_{prediction_horizons[idx_len_prediction]}.png")
                    plt.close()
        # ------------------------------ THIS IS NECESSARY -------------------------------
                    
            list_of_R2.append(r_squared_mean_along_y_dimensions[:,0])
            list_of_RMSE.append(RMSE_mean_along_y_dimensions[:,0])

        # --------------------- RECAP OF SINGLE MODEL (R2_RMSE) -------------------------------
            
        num_tests = np.arange(0,len(list_of_available_tensors))
        tickvalues = range(1,len(list_of_available_tensors)+1) 
        list_labels = []
        for k in range(1,len(list_of_available_tensors)+1): 
            list_labels.append( "test_" + str(k-1))

        font = {'family' : 'DejaVu Sans',
                'weight' : 'normal',
                'size'   : 25}
        plt.rc('font', **font)

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18,18),gridspec_kw={'height_ratios': [3.2, 1, 3.2]})
        fig.suptitle(f'Metrics - model: {model_name} - Prediction {prediction_horizons[idx_len_prediction]}',fontsize=25)
        #R2
        axes[0].plot(torch.zeros(len(tickvalues)+2),'k--',linewidth=1.5,alpha=0.5)
        axes[0].plot(torch.ones(len(tickvalues)+2),'b--',linewidth=1.5,alpha=0.5)
        axes[0].set_title('$R^2$', y=1.02,size=25)
        axes[0].boxplot(list_of_R2,1,'',patch_artist = True, boxprops = dict(facecolor = "lightblue"), medianprops = dict(color = "green", linewidth = 1.5), whiskerprops = dict(color = "red", linewidth = 2))
        axes[0].set_xlim(0.5,len(list_of_available_tensors)+0.5)

        axes[0].grid(True,alpha=0.6)    
        #RMSE
        axes[2].plot(torch.zeros(len(tickvalues)+2),'k--',linewidth=1.5,alpha=0.5)
        axes[2].set_title('$RMSE$', y=1.02,size=25)
        axes[2].boxplot(list_of_RMSE,1,'',patch_artist = True, medianprops = dict(color = "green", linewidth = 1.5), boxprops = dict(facecolor = "lightblue"), whiskerprops = dict(color = "red", linewidth = 2))
        axes[2].set_xlim(0.5,len(list_of_available_tensors)+0.5)
        axes[2].grid(True,alpha=0.6)   

        axes[1].axis('off')

        # For all the tested models, print a recap of 
        for j in range(len(num_tests)):
            first_idx = list_of_available_models[model_idx].find("_ds")
            last_idx = list_of_available_models[model_idx].find(".pt")
            dataset_name = list_of_available_models[model_idx][first_idx:last_idx]
            training_list_file = f"training{dataset_name}_list.txt"
            training,test,output,boolean_output = check_ID_OOD(list_of_available_tensors[j], 
                                                            training_list_file)
            if any(map(lambda elem: elem is None, (training,test,output,boolean_output))):
                output = "Error_validation"
            else:
                differing_classes = boolean_output.count(False)

            if  output == "Error_validation":
                bbox_dict =dict(facecolor = 'red', alpha = 0.5)
                # programPause = input("Error In this file...")
                suptitle = "\nValidation"
            elif output == "In-Distribution":
                bbox_dict = dict(facecolor = 'green', alpha = .4)
                suptitle = f"\nID\n{test['task']}"
            elif output == "Out-Of-Distribution":
                bbox_dict =dict(facecolor = 'orange', alpha = .4)
                suptitle = f"\nOOD ({differing_classes} cat)\n{test['task']} "  #  

            if len(num_tests) == 8:
                pos = (j+ (j+1))/2 -.5
            else:
                pos = (j+ (j+1))/1.9 -.5    
            axes[1].text( pos ,0 , '$R^2$: ' +str(round(test_metrics[idx_len_prediction,j,0].item(),3))+'\n$\sigma_{R^{2}}$: ' +
                    str(round(test_metrics[idx_len_prediction,j,1].item(),3))+'\n$RMSE$: ' +str(round(test_metrics[idx_len_prediction,j,2].item(),4))+'\n'+'$\sigma_{RMSE}$: ' +
                    str(round(test_metrics[idx_len_prediction,j,4].item(),4))+ suptitle, fontsize = 16, bbox = bbox_dict  ) 

        axes[1].set_ylim(0, .5)

        plt.setp(axes, xticks=tickvalues, xticklabels=list_labels)
        plt.tight_layout()

        if save_figure == True:
            output_name = f"R2_RMSE_{str(model_name).replace('.pt','')}_horizon_{str(prediction_horizons[idx_len_prediction])}.png"
            plt.savefig(model_path / output_name ) 
            axes[0].set_ylim(-0.2,1.1)
            axes[2].set_ylim(0,0.2)
            output_name = f"R2_RMSE_{str(model_name).replace('.pt','')}_horizon_{str(prediction_horizons[idx_len_prediction])}_fixed_scale.png"
            plt.tight_layout( pad = 1.2)
            plt.savefig(model_path / output_name )
            plt.close()
        else:
            plt.show()        

        list_of_RMSE_total.append(list_of_RMSE)
        list_of_R2_total.append(list_of_R2)
       
    
    # ----------R2 and RMSE for single test all predictions ---------------
    
    test_idx = 0
    tickvalues = range(1,len(prediction_horizons)+1) 
    list_labels = []
    for k in range(1,len(prediction_horizons)+1): 
        list_labels.append( "pred_" + str(prediction_horizons[k-1]))

    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 30}
    plt.rc('font', **font)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18,14),gridspec_kw={'height_ratios': [3 ,3]})
    fig.suptitle(f'Metrics on test_{test_idx} - context {cfg.seq_len_ctx} steps - All horizons' ,fontsize=30, y =0.96)
    
    axes[0].plot(torch.zeros(len(tickvalues)+2),'k--',linewidth=2,alpha=0.8)
    axes[0].plot(torch.ones(len(tickvalues)+2),'b--',linewidth=2,alpha=0.8)
    axes[0].set_title('$R^2$', y=1.02,size=25)
    axes[1].plot(torch.zeros(len(tickvalues)+2),'k--',linewidth=2,alpha=0.8)
    axes[1].set_title('$RMSE$', y=1.02,size=25)

    for k in range(len(list_of_R2_total)):
    #R2
        axes[0].boxplot(list_of_R2_total[k][test_idx],1,'',meanline=True, showmeans=True,  positions=[k+1] ,patch_artist = True, boxprops = dict(facecolor = "lightblue"),meanprops = dict(color = "magenta", linewidth = 1.5), medianprops = dict(color = "green", linewidth = 1.5), whiskerprops = dict(color = "red", linewidth = 2))
    #RMSE
        axes[1].boxplot(list_of_RMSE_total[k][test_idx],1,'',meanline=True, showmeans=True, positions=[k+1], patch_artist = True, meanprops = dict(color = "magenta", linewidth = 1.5), medianprops = dict(color = "green", linewidth = 1.5), boxprops = dict(facecolor = "lightblue"), whiskerprops = dict(color = "red", linewidth = 2))
    
    axes[0].set_xlim(0.5,len(prediction_horizons)+0.5)    
    axes[1].set_xlim(0.5,len(prediction_horizons)+0.5)    
    axes[1].grid(True,alpha=0.6)   
    axes[0].grid(True,alpha=0.6)    

    plt.setp(axes, xticks=tickvalues, xticklabels=list_labels)
    plt.tight_layout()
    # plt.show()
    if save_figure == True:
        output_name = f"R2_RMSE_{str(model_name).replace('.pt','')}_ALL_horizons_test{test_idx}.png"
        plt.savefig(fig_path / output_name ) 
        axes[0].set_ylim(-0.2,1.1)
        axes[1].set_ylim(0,0.2)
        output_name = f"R2_RMSE_{str(model_name).replace('.pt','')}_ALL_horizons_test{test_idx}_FS.png"
        plt.tight_layout( pad = 1.2)
        plt.savefig(fig_path / output_name )
        plt.close()
    else:
        plt.show()        

   # --------------------------R2 of all horizons for a single model ---------------------------------------         
    
    num_tests = np.arange(0,len(list_of_available_tensors))

    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 25}
    plt.rc('font', **font)

    for i in range(len(prediction_horizons)):    

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18,15),gridspec_kw={'height_ratios': [6, 1]})

        axes[1].axis('off')
        plt.suptitle('Model '+ str(list_of_available_models[model_idx]) + " - Prediction " 
                     + str(prediction_horizons[i]) +' steps',fontsize=25)
        
        axes[0].errorbar(num_tests, test_metrics[i,:,0].numpy(),fmt="o",markersize='20', yerr=test_metrics[i,:,1], 
                    ecolor= 'red', elinewidth = 3,capthick=3, capsize=20)
        axes[0].plot(torch.zeros(len(num_tests)),'k--',linewidth=2)
        axes[0].plot(torch.ones(len(num_tests)),'b--',linewidth=2)

        # xlabel creation
        # Set names for x_axis
        tickvalues = range(0,len(list_of_available_tensors)) 
        list_labels = []

        for k in range(len(list_of_available_tensors)): 
            list_labels.append( "test_" + str(k))
        
        # axes[0].set_yticks(fontsize=25)
        axes[0].set_title('$R^{2}$',size=25,rotation=0)
        # axs.set_ylim([-1.2, 1.5])
        axes[0].grid(True,alpha=0.7)
        
        for j in range(len(num_tests)):

            first_idx = list_of_available_models[model_idx].find("_ds")
            last_idx = list_of_available_models[model_idx].find(".pt")
            dataset_name = list_of_available_models[model_idx][first_idx:last_idx]
            training_list_file = f"training{dataset_name}_list.txt"
            training,test,output,boolean_output = check_ID_OOD(list_of_available_tensors[j], 
                                                            training_list_file)
            if any(map(lambda elem: elem is None, (training,test,output,boolean_output))):
                output = "Error_validation"
            else:
                differing_classes = boolean_output.count(False)

            if  output == "Error_validation":
                bbox_dict =dict(facecolor = 'red', alpha = 0.5)
                suptitle = "\nValidation"
            elif output == "In-Distribution":
                bbox_dict = dict(facecolor = 'green', alpha = .4)
                suptitle = f"\nID\n{test['task']}"
            elif output == "Out-Of-Distribution":
                bbox_dict =dict(facecolor = 'orange', alpha = .4)
                suptitle = f"\nOOD ({differing_classes} cat)\n{test['task']} "  #  

            axes[1].text((j+ (j+1))/2.2 -.6, 0, '$R^2$: ' +str(round(test_metrics[i,j,0].item(),4))+'\n'+'$\sigma_{R^{2}}$: ' +
                        str(round(test_metrics[i,j,1].item(),4))+suptitle, fontsize = 16, bbox = bbox_dict  ) 
        axes[1].set_ylim(0, .5)

        plt.setp(axes, xticks=tickvalues, xticklabels=list_labels)
        plt.tight_layout()

        if save_figure == True:

            output_name = f"benchmark_{str(list_of_available_models[model_idx]).replace('.pt','')}_horizon{prediction_horizons[i]}"
            output_name2 = f"{output_name}.png"
            plt.tight_layout()
            plt.savefig(model_path / output_name2 )
            # -------------------------------------------------------------------
            output_name2 = f"{output_name}_fixed_scale.png"
            axes[0].set_ylim(-0.2,1.1)

            plt.savefig(fig_path / output_name2 )       
            plt.close()
        else:
            plt.show()

        # ----------------------------- RMSE -------------------------------------------------

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18,15),gridspec_kw={'height_ratios': [6, 1]})
        axes[1].axis('off')
        
        plt.suptitle('Model '+ str(list_of_available_models[model_idx]) + " - Prediction " 
                + str(prediction_horizons[i]) +' steps',fontsize=25)
        
        axes[0].errorbar(num_tests, test_metrics[i,:,2],fmt="o",markersize='20', yerr=test_metrics[i,:,3], 
                    ecolor= 'red', elinewidth = 3,capthick=3, capsize=20)
        axes[0].plot(torch.zeros(num_tests.shape),'k--',linewidth=2)

        # Set names for x_axis
        tickvalues = range(0,len(list_of_available_tensors)) 
        list_labels = []

        for k in range(len(list_of_available_tensors)): 
            list_labels.append( "test_" + str(k))
        
        # axes[0].set_yticks(fontsize=25)
        axes[0].set_title('$RMSE$',size=25,rotation=0)
        # axs.set_ylim([-1.2, 1.5])
        axes[0].grid(True,alpha=0.7)

        for j in range(len(num_tests)):

            first_idx = list_of_available_models[model_idx].find("_ds")
            last_idx = list_of_available_models[model_idx].find(".pt")
            dataset_name = list_of_available_models[model_idx][first_idx:last_idx]
            training_list_file = f"training{dataset_name}_list.txt"
            training,test,output,boolean_output = check_ID_OOD(list_of_available_tensors[j], 
                                                            training_list_file)
            if any(map(lambda elem: elem is None, (training,test,output,boolean_output))):
                output = "Error_validation"
            else:
                differing_classes = boolean_output.count(False)
            if  output == "Error_validation":
                bbox_dict =dict(facecolor = 'red', alpha = 0.5)
                suptitle = "\nValidation"
            elif output == "In-Distribution":
                bbox_dict = dict(facecolor = 'green', alpha = .4)
                suptitle = f"\nID\n{test['task']}"
            elif output == "Out-Of-Distribution":
                bbox_dict =dict(facecolor = 'orange', alpha = .4)
                suptitle = f"\nOOD ({differing_classes} cat)\n{test['task']} "  #  

            axes[1].text((j+ (j+1))/2.2 -.6, 0 , '$RMSE$: ' +str(round(test_metrics[i,j,2].item(),4))+'\n'+'$\sigma_{RMSE}$: ' +
                    str(round(test_metrics[i,j,3].item(),4))+suptitle, fontsize = 16, bbox = bbox_dict  ) 
        axes[1].set_ylim(0, .5)

        plt.setp(axes, xticks=tickvalues, xticklabels=list_labels)
        plt.tight_layout()
        if save_figure == True:
            output_name = f"benchmark_RMSE_{str(list_of_available_models[model_idx]).replace('.pt','')}_horizon{prediction_horizons[i]}"
            
            plt.savefig(model_path / output_name )
            # -------------------------------------------------------------------
            output_name = f"{output_name}_fixed_scale.png"
            axes[0].set_ylim(bottom = -0.02 )
            plt.savefig(model_path / output_name )       
            plt.close()
        else:
            plt.show()

    # --------------------------- TABLE RECAP -------------------------------------------
    
    from tabulate import tabulate

    torch.save(test_metrics,model_path / f'metrics_model_{list_of_available_models[model_idx][:-3]}.pt')            

    f = open( model_path /"Recap_model_among_tests.txt", "w")
    col_names = ['R2','stdR2','RMSE','stdRMSE','NRMSE', 'stdNRMSE','Fit_index','stdFit_index']

    print(f"Model--> "+ list_of_available_models[model_idx])
    f.write(f"Model--> "+ list_of_available_models[model_idx])

    for l in range(len(prediction_horizons)):
        print(f"Horizon{l} --> "+ str(prediction_horizons[l]))
        f.write(f"Horizon{l} --> "+ str(prediction_horizons[l]))
        print("\n")
        f.write("\n")

    for j in range(len(list_of_available_tensors)):
        print(f"Test_file{j} --> "+ list_of_available_tensors[j])
        f.write(f"Test_file{j} --> "+ list_of_available_tensors[j])
        print("\n")
        f.write("\n")

    for i in range(len(prediction_horizons)):
        print("\nMODEL: "+ list_of_available_models[model_idx])
        f.write("\nMODEL: "+ list_of_available_models[model_idx]+"\n\n")
        print("Prediction Horizon : "+ str(prediction_horizons[i]))
        f.write("Prediction Horizon: "+ str(prediction_horizons[i])+"\n\n")
        print(tabulate(test_metrics[i], headers=col_names))
        f.write(tabulate(test_metrics[i], headers=col_names, tablefmt="fancy_grid"))

    f.close()

log.close()