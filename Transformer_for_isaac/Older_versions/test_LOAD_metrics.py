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
threads = 5
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


model_path = os.path.abspath("./out")
list_of_available_models = os.listdir(model_path)

#------------------------------------------------------------------------------------------
test_metrics = torch.load("./fig/metrics_models.pt")

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

    for j in range(len(x)):

        training_frequency = 0.15 #extract from the file !!! WIP 
        training_frequency = str(training_frequency).replace('.','_')
        mass_bounds = 10,10
        mass_bounds = str(mass_bounds).replace('(','').replace(')','').replace(', ','_')

        current_test = list_of_available_tensors[j]
        frequency_index = current_test.find('f_')
        randomization_index = current_test.find('mass_') 
        # --------------------------------------------------------------------------
        if current_test[frequency_index+2:frequency_index+6] == str(training_frequency):
            suptitle1 = ''
            flag1 = True
        else:
            suptitle1 = '\n f: ' + current_test[frequency_index+2:frequency_index+5].replace('_','.')+' Hz'
            flag1 = False

        if current_test[randomization_index+5:randomization_index+10].replace('p','') == mass_bounds:
            suptitle2 = ''
            flag2 = True
        else:
            suptitle2 = '\n +' +current_test[randomization_index+5:randomization_index+10].replace('_','/-').replace('.p','') +' % '
            flag2 = False

        suptitle = suptitle1 + suptitle2 
        flag = flag1 and flag2
        if flag:
            bbox_dict = dict(facecolor = 'green', alpha = 0.4)
        else:
            bbox_dict =dict(facecolor = 'red', alpha = 0.7)

        plt.text(j-0.3, -0.7, '$R^2$: ' +str(round(test_metrics[i,j,0].item(),4))+'\n'+'$\sigma_{R^{2}}$: ' +
                 str(round(test_metrics[i,j,1].item(),4))+suptitle, fontsize = 18, bbox = bbox_dict  ) 

    if save_figure == True:
        output_name = f"benchmark_{str(list_of_available_models[i]).replace('.pt','')}.png"
        plt.tight_layout()
        plt.savefig(fig_path / output_name )
    else:
        plt.show()
