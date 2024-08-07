from pathlib import Path
import time
import torch
import numpy as np
import math
from functools import partial
# from dataset import WHDataset, LinearDynamicalDataset
from torch.utils.data import DataLoader
from transformer_sim import Config, TSTransformer
from Older_versions.transformer_onestep import warmup_cosine_lr
import tqdm
import argparse
import wandb
from datetime import datetime
import re
# from torch.utils.tensorboard import Summary# writer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meta system identification with transformers - IsaacGym')

    # Overall
    parser.add_argument('--train-folder', type=str, default="train", metavar='S',
                        help='train folder')
    parser.add_argument('--model-dir', type=str, default="out", metavar='S',
                        help='Saved model folder')
    parser.add_argument('--out-file', type=str, default="ckpt", metavar='S',
                        help='Saved model name')
    parser.add_argument('--in-file', type=str, default="ckpt", metavar='S',
                        help='Loaded model name (when resuming)')
    parser.add_argument('--init-from', type=str, default="scratch", metavar='S',
                        help='Init from (scratch|resume|pretrained)')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='Seed for random number generation')
    parser.add_argument('--log-wandb', default=False,     # --log-wandb', action='store_true', default=False,
                        help='Live log')

    # Dataset
    parser.add_argument('--nx', type=int, default=7, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--nu', type=int, default=7, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--ny', type=int, default=14, metavar='N',
                        help='model order (default: 5)')
    
    parser.add_argument('--seq-len-ctx', type=int, default=400, metavar='N',
                        help='sequence length (default: 300)')
    parser.add_argument('--seq-len-new', type=int, default=400, metavar='N',
                        help='sequence length (default: 300)')
    parser.add_argument('--mag_range', type=tuple, default=(0.5, 0.97), metavar='N',
                        help='sequence length (default: 600)')
    parser.add_argument('--phase_range', type=tuple, default=(0.0, math.pi/2), metavar='N',
                        help='sequence length (default: 600)')
    parser.add_argument('--fixed-system', action='store_true', default=False,
                        help='If True, keep the same model all the times')

    # Model
    parser.add_argument('--n-layer', type=int, default=12, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--n-head', type=int, default=4, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--n-embd', type=int, default=128, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--bias', action='store_true', default=True,
                        help='bias in model')

    # Training
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--max-iters', type=int, default=1_000_000, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--warmup-iters', type=int, default=10_000, metavar='N',
                        help='number of iterations (default: 1000)')
    parser.add_argument('--lr', type=float, default=6e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='D',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--eval-interval', type=int, default=100, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--eval-iters', type=int, default=50, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--fixed-lr', action='store_true', default=False,
                        help='disables CUDA training')

    # Compute
    parser.add_argument('--threads', type=int, default=10,
                        help='number of CPU threads (defa400ult: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda-device', type=str, default="cuda:0", metavar='S',
                        help='cuda device (default: "cuda:0")')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='disables CUDA training')
    
    # ------------------------------------ Added ---------------------------------------------

    parser.add_argument('--custom-dataset', action='store_true', default=True,
                    help='Dataset from Isaac or simple sys')
    parser.add_argument('--manuel_pc',default=True,   
                    help='if you are Training on lab --> False')
    parser.add_argument('--context', type=float, default=.2,
                help='How much of the timesteps is context')
    parser.add_argument('--loss-function', type=str, default='MSE',
                help="Loss function: 'MAE'-'MSE'-'Huber'")
    parser.add_argument('--iter-log', type=int, default=100,
                        help='Iteration every which logs wandb')
    
    # ------------------------------------ Edited ------------------------------------
    parser.add_argument('--beta1', type=float, default=.9,
                help="Loss function: 'MAE'-'MSE'-'Huber'")
    parser.add_argument('--beta2', type=float, default=.95,
                help="Loss function: 'MAE'-'MSE'-'Huber'")
    
    cfg,unparsed = parser.parse_known_args() #cfg = parser.parse_args()

    partition = cfg.context # % of context respect prediction

    # Derived settings
    #cfg.block_size = cfg.seq_len
    cfg.lr_decay_iters = cfg.max_iters
    cfg.min_lr = cfg.lr/10.0  #
    cfg.decay_lr = not cfg.fixed_lr
    cfg.eval_batch_size = cfg.batch_size

    current_time = re.sub(":|-|\s", "_", str(datetime.now())[5:16])
    current_time = 'DAY_'+current_time[:5]+'|TIME_'+current_time[6:]
    
    # if cfg.manuel_pc:               #setting up tensorboard

    #     # print(current_time)
    #     # cfg.batch_size = 8

    # writer = Summary# writer(log_dir=f"runs/{current_time}_RUN_nu{cfg.nu}_ny{cfg.ny}_partition_{partition}_batch{cfg.batch_size}_nEmbd{cfg.n_embd}")

    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed) # not needed? All randomness now handled with generators

    # Create out dir
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(exist_ok=True)

    # Configure compute
    torch.set_num_threads(cfg.threads)
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device_name = cfg.cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = 'cuda' if 'cuda' in device_name else 'cpu'

    # if cfg.manuel_pc ==str(False):       # it means that training is on lab, so it should support it
    #     torch.set_float32_matmul_precision("high")

        # ---------------------------------------------------------------------------------------------------

    if cfg.custom_dataset == False:
        print("custom dataset false")
        # Init wandb
        if cfg.log_wandb:
            wandb.init(
                project="sysid-meta-Isaac-gym-prove-normalization",
                name=f"partition_{round(partition*100)}_batch{cfg.batch_size}_nEmbd{cfg.n_embd}",
                config=vars(cfg))
            
        # Create data loader
        train_ds = LinearDynamicalDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, seq_len=cfg.seq_len_ctx+cfg.seq_len_new)

        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.threads)

        val_ds = LinearDynamicalDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, seq_len=cfg.seq_len_ctx+cfg.seq_len_new)
        val_dl = DataLoader(val_ds, batch_size=cfg.eval_batch_size, num_workers=cfg.threads)

    else:

        # ---------------------------------------------------------------------------------------------------

        from torch.utils.data import random_split
        import os   

        parent_folder = os.path.join(os.getcwd(), os.pardir) 
        parent_folder = os.path.abspath(parent_folder)

        print("cfg.manuel_pc --> ",cfg.manuel_pc)
        # different names for master folder
        # if cfg.manuel_pc == str(True):
        #     # import ipdb
        #     # ipdb.set_trace()
        #     print("------------ Working on Manuel's PC ---------------- ")
        #     relative_folder = "isaacgym/python/examples/Franka/out_tensors/train"
        # else:
        #     print("------------ Working on Lab's PC ------------------")
        #     relative_folder = "Data_generation/python/examples/Franka/out_tensors/train"

        try:
            relative_folder = f"./data_generation/out_tensors/{cfg.train_folder}"
            tensors_path = os.path.join(parent_folder,relative_folder ) 
            tensors_path = os.path.abspath(tensors_path)
            list_of_available_tensors = os.listdir(tensors_path)
        except FileNotFoundError:
            relative_folder = f"Data_generation/python/examples/Franka/out_tensors/{cfg.train_folder}"
            tensors_path = os.path.join(parent_folder,relative_folder ) 
            tensors_path = os.path.abspath(tensors_path)
            list_of_available_tensors = os.listdir(tensors_path)

        tensors_path = os.path.join(parent_folder,relative_folder ) 
        tensors_path = os.path.abspath(tensors_path)
        directory_available_tensors = os.listdir(tensors_path)

        print("There are "+str(len(directory_available_tensors)) +" available files in "+str(tensors_path) +":\n")

        list_available_tensors =[]
        for i in range(len(directory_available_tensors)):
            single_pt_file_path = os.path.join(tensors_path, directory_available_tensors[i]) 
            list_available_tensors.append(directory_available_tensors[i])
            print(" --> ",directory_available_tensors[i])

        # -------------------------------------------------------------------------------------------------------- # 

        time_start = time.time()

        number_of_pt_files = len(directory_available_tensors)

        tensors_used_for_training = list_available_tensors[0:number_of_pt_files-1]

        #This is the training loop
        for i in range(number_of_pt_files):

            single_pt_file_path = os.path.join(tensors_path, directory_available_tensors[i]) 
            print("\n\n ----------------------------------\n") 
            print("Starting with file:\n", single_pt_file_path)
            
            loaded = torch.load(single_pt_file_path,map_location=device) #

            @ torch.no_grad()
            def loading():
                control_action_extracted = loaded['control_action'][1:,:,:7]
                position_extracted = loaded['position']
                position = torch.movedim(position_extracted.to('cpu'),-2,-3)
                control_action = torch.movedim(control_action_extracted.to('cpu'),-2,-3)
                
                control_action = control_action[:,:,:7] # first is the number of simulations
                position = position[:,:,:]
                return control_action,position

            control_action,position = loading()
            print(control_action.shape)
            print(position.shape)

            train_dataset = torch.utils.data.TensorDataset(position, control_action)
            split_ratio = 0.8
            train_size = int(split_ratio * len(train_dataset))
            valid_size = len(train_dataset) - train_size
            train_ds, val_ds = random_split(train_dataset, [train_size, valid_size])

            train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.threads)
            val_dl = DataLoader(val_ds, batch_size=cfg.eval_batch_size, num_workers=cfg.threads)
            
            cfg.ny = position.shape[2]
            cfg.nu = control_action.shape[2]
            cfg.seq_len = position.shape[1]
            cfg.block_size = cfg.seq_len
            # partition = np.random.randint(1,20)/100
            # print(partition)
            cfg.seq_len_ctx = int(partition * cfg.seq_len)
            cfg.seq_len_new = cfg.seq_len - cfg.seq_len_ctx
            cfg.in_file = f"{cfg.out_file}_partition_{round(partition*100)}_batch{cfg.batch_size}_embd{cfg.n_embd}_heads{cfg.n_head}_lay{cfg.n_layer}_{cfg.loss_function}"
            # cfg.in_file = "ckpt_partition_20_batch16_embd192_heads12_lay12_MSE_ds_big_ft4_FC_rand"
            # 
            ckpt_path = model_dir / f"{cfg.in_file}.pt"
            ckpt_is_present = os.path.isfile(ckpt_path)

            if ckpt_is_present == False:
                print(f"{cfg.in_file}.pt" + " is not present --> Proceeding from scratch") 
                cfg.init_from = "scratch"
            else:
                print(f"{cfg.in_file}.pt" + " is present --> Proceeding from resume") 
                cfg.init_from = "resume"

            if cfg.log_wandb and i == 0: #initialize only at the first iteration
                wandb.init(
                group='Ds_1',
                project="sysid-Franka",
                name = f"batch{cfg.batch_size}_embd{cfg.n_embd}_lay{cfg.n_layer}_heads{cfg.n_head}_{cfg.loss_function}",
                config=vars(cfg))
            
    # ---------------------------------------------------------------------------------------------------

            model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, n_y=cfg.ny, n_u=cfg.nu,
                            seq_len_ctx=cfg.seq_len_ctx, seq_len_new=cfg.seq_len_new,
                            bias=cfg.bias, dropout=cfg.dropout)  
            
            if cfg.init_from == "scratch":
                gptconf = Config(**model_args)
                model = TSTransformer(gptconf)
            elif cfg.init_from == "resume" or cfg.init_from == "pretrained":
                ckpt_path = model_dir / f"{cfg.in_file}.pt"
                checkpoint = torch.load(ckpt_path, map_location=device)
                gptconf = Config(**checkpoint["model_args"])
                model = TSTransformer(gptconf)
                state_dict = checkpoint['model']

                model.load_state_dict(state_dict)
            model.to(device)

            if cfg.compile:
                model = torch.compile(model)  # requires PyTorch 2.0
            optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)

            if cfg.init_from == "resume":
                optimizer.load_state_dict(checkpoint['optimizer'])

            @torch.no_grad()
            def estimate_loss():
                model.eval()
                loss = 0.0
                for eval_iter, (batch_y, batch_u) in enumerate(val_dl):
                    if device_type == "cuda":
                        batch_y = batch_y.pin_memory().to(device, non_blocking=True)
                        batch_u = batch_u.pin_memory().to(device, non_blocking=True)
                    #_, loss_iter = model(batch_u, batch_y)
                    
                    batch_y_ctx = batch_y[:, :cfg.seq_len_ctx, :]
                    batch_u_ctx = batch_u[:, :cfg.seq_len_ctx, :]
                    batch_y_new = batch_y[:, cfg.seq_len_ctx:, :]
                    batch_u_new = batch_u[:, cfg.seq_len_ctx:, :]
                    batch_y_sim = model(batch_y_ctx, batch_u_ctx, batch_u_new)
                    
                    # Original
                    if cfg.loss_function =='MSE':
                        loss_iter = torch.nn.functional.mse_loss(batch_y_new, batch_y_sim)
                    elif cfg.loss_function =='MAE':
                        loss_iter = torch.nn.functional.l1_loss(batch_y_new, batch_y_sim)
                    # huber loss
                    elif cfg.loss_function =='Huber':
                        loss_iter = torch.nn.functional.huber_loss(batch_y_new, batch_y_sim, delta = 0.2) 
                    else :
                        print("\nError, loss function --> Break.")   
                        break                 
                    #loss_iter = torch.nn.functional.mse_loss(batch_y_new[:, 1:, :], batch_y_sim[:, :-1, :])

                    loss += loss_iter.item()
                    if eval_iter == cfg.eval_iters:
                        break
                loss /= cfg.eval_iters
                model.train()
                return loss

            # Training loop
            LOSS_ITR = []
            LOSS_VAL = []
            loss_val = np.nan

            if cfg.init_from == "scratch" or cfg.init_from == "pretrained":
                iter_num = 0
                best_val_loss = np.inf
            elif cfg.init_from == "resume":
                iter_num = checkpoint["iter_num"]
                best_val_loss = checkpoint['best_val_loss']

            get_lr = partial(warmup_cosine_lr, lr=cfg.lr, min_lr=cfg.min_lr,
                            warmup_iters=cfg.warmup_iters, lr_decay_iters=cfg.lr_decay_iters)

            for iter_num, (batch_y, batch_u) in tqdm.tqdm(enumerate(train_dl, start=iter_num)):

                if (iter_num % cfg.eval_interval == 0) and iter_num > 0:
                    loss_val = estimate_loss()
                    LOSS_VAL.append(loss_val)
                    print(f"\n{iter_num=} {loss_val=:.4f}\n")
                    if loss_val < best_val_loss:
                        best_val_loss = loss_val
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'train_time': time.time() - time_start,
                            'LOSS': LOSS_ITR,
                            'LOSS_VAL': LOSS_VAL,
                            'best_val_loss': best_val_loss,
                            'cfg': cfg,
                        }
                        # torch.save(checkpoint, model_dir / f"{cfg.out_file}.pt")
                
                
                # determine and set the learning rate for this iteration
                if cfg.decay_lr:
                    lr_iter = get_lr(iter_num)
                else:
                    lr_iter = cfg.lr

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_iter

                if device_type == "cuda":
                    batch_y = batch_y.pin_memory().to(device, non_blocking=True)
                    batch_u = batch_u.pin_memory().to(device, non_blocking=True)

                ########### Normalization of input ###### CHECK
            # batch_y, batch_u -->

                # batch_u = (batch_u - batch_u.mean(axis=0)) / (batch_u.std(axis=0) + 1e-6)

                mean_u = batch_u.mean(axis=1, keepdim=True)
                std_u = batch_u.std(axis=1, keepdim=True)

                # Normalizza il tensore
                batch_u = (batch_u - mean_u) / (std_u + 1e-6)

                mean_y = batch_y.mean(axis=1, keepdim=True)
                std_y = batch_y.std(axis=1, keepdim=True)

                # Normalizza il tensore
                batch_y = (batch_y - mean_y) / (std_y + 1e-6)

                if iter_num % cfg.iter_log == 0:
                    print(batch_u.shape)
    
                #  u = u.astype(self.dtype) (guarda in dataset.py)
                
                # test normalizzare solo input, dopodichè sia input che output
                # salvare mean e std di ogni batch e utilizzarlo per la formula inversa

                batch_y_ctx = batch_y[:, :cfg.seq_len_ctx, :]
                batch_u_ctx = batch_u[:, :cfg.seq_len_ctx, :]
                batch_y_new = batch_y[:, cfg.seq_len_ctx:, :]
                batch_u_new = batch_u[:, cfg.seq_len_ctx:, :]

                batch_y_sim = model(batch_y_ctx, batch_u_ctx, batch_u_new)

                if cfg.loss_function =='MSE':
                    loss = torch.nn.functional.mse_loss(batch_y_new, batch_y_sim)
                elif cfg.loss_function =='MAE':
                    loss = torch.nn.functional.l1_loss(batch_y_new, batch_y_sim)
                    # huber loss
                elif cfg.loss_function =='Huber':
                    loss = torch.nn.functional.huber_loss(batch_y_new, batch_y_sim, delta = 0.2) 
                else :
                    print("\nError, loss function --> Break.")

                #loss = torch.nn.functional.mse_loss(batch_y_new[:, 1:, :], batch_y_sim[:, :-1, :])

                LOSS_ITR.append(loss.item())

                # print every 100 iters
                if iter_num % cfg.iter_log == 0:
                    print(f"\n{iter_num=} {loss=:.4f} {loss_val=:.4f} {lr_iter=}\n")
                    # writer.add_scalar('Train Loss', loss, iter_num) #added
                    # writer.add_scalar('Val Loss', loss_val, iter_num)
   
                    if cfg.log_wandb:
                        wandb.log({"loss": loss, "loss_val": loss_val})

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if iter_num == cfg.max_iters-1:
                    break

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'train_time': time.time() - time_start,
                'LOSS': LOSS_ITR,
                'LOSS_VAL': LOSS_VAL,
                'best_val_loss': best_val_loss,
                'cfg': cfg,
            }
            torch.save(checkpoint, model_dir / f"{cfg.out_file}_partition_{round(partition*100)}_batch{cfg.batch_size}_embd{cfg.n_embd}_heads{cfg.n_head}_lay{cfg.n_layer}_{cfg.loss_function}.pt")
            # 
        # -------------------------------------------------------------------------------------------------
        
        metric_dict = dict(iter_num = iter_num, train_time =( time.time() - time_start),
                            batch_size = cfg.batch_size, best_val_loss = best_val_loss)
        
        # # writer.add_hparams(model_args,metric_dict)
        # # writer.add_text("trained on:",''.join(tensors_used_for_training).replace('.pt','.pt ---- '),0)     
        # # writer.flush()
        # # writer.close()


        # These lines were moved because training time need to account for all the iterations!

        if cfg.log_wandb:
            wandb.finish()
        time_loop = time.time() - time_start
        print(f"\n{time_loop=:.2f} seconds.")