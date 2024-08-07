source "$HOME/miniconda3/etc/profile.d/conda.sh"

for j in $(seq 1 10); do

    #Move to Franka folder
    conda activate rlgpu

    cd ../data_generation
    
    # delete train directory
    rm -r ./out_tensors/train

    for i in $(seq 5); do
        for frequency in 0.1 0.15 0.2; do
            for bounds in 10; do   
                for type_input in 'multi_sinusoidal'; do                                                                                                                               
                    python generation_franka.py --num-envs 1024 --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input "${type_input}" --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test' 
                done
            done
        done
    done

    # save checkpoint of trained values
    cp ./training_dataset_list.txt ./training_dataset_list${j}.txt

    # Move to Transformers
    cd ../Transformer_for_isaac

    # Activate conda environment for training 
    conda activate RoboMorph

    #Trainining
    for batches in 16; do
        for embd in 192; do
            for head in 12; do
                for layers in 12; do
                    for partition in  .5; do             
                        for loss in 'MSE'; do                                                                                                                       
                            python train_sim_multiple_seeds.py --log-wandb True --train-folder "train" 
                            --model-dir "Normalized_u" --bias --loss-function "${loss}" --log-wandb True 
                            --context "${partition}" --n-layer "${layers}" --n-head "${head}" --n-embd "${embd}" 
                            --batch-size "${batches}" --manuel_pc False   
                        done
                    done
                done
            done
        done
    done
    cp -r ./Normalized_u Normalized_u"${j}"
done

# Go in franka and duplicate file .txt from no_tuning
cd ../data_generation
rm training_dataset_list.txt
cp ./training_dataset_list_no_ft.txt ./training_dataset_list.txt

cd ../Transformer_for_isaac

# cp -r Prova_big_training_NO_ft Normalized_u_FC


