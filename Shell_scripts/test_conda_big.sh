source "$HOME/miniconda3/etc/profile.d/conda.sh"

for j in $(seq 1 4); do

    #Move to Franka folder
    conda activate rlgpu

    cd ../Data_generation/python/examples/Franka
    
    # delete train directory
    rm -r ./out_tensors/train

    for i in $(seq 1 100); do
        for type in 'VS'; do # 
            for bounds in 10; do                                                                                                                                 
                python scratch_2024b.py --num-envs 4096 --random-stiffness-dofs False 
                --disable-gravity False --control-imposed-file False --control-imposed False --osc-task True 
                --type-of-task "${type}" --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'train'
            done
        done
    done

    # save checkpoint of trained values
    cp ./training_dataset_list.txt ./training_dataset_list${j}.txt

    # Move to Transformers
    cd ../../../../Transformer_for_isaac

    # Activate conda environment for training 
    conda activate robomorph
    #Trainining
    for batches in 16; do
        for embd in 192; do
            for head in 12; do
                for layers in 12; do
                    for partition in  .2; do             
                        for loss in 'MSE'; do                                                                                                                       
                            python train_sim_multiple_seeds.py --log-wandb True --train-folder "train" 
                            --model-dir "Prova_big_training_ft" --bias --loss-function "${loss}" --log-wandb True 
                            --context "${partition}" --n-layer "${layers}" --n-head "${head}" --n-embd "${embd}" 
                            --batch-size "${batches}" --manuel_pc False   
                        done
                    done
                done
            done
        done
    done
    cp -r ./Prova_big_training_ft/ Prova_big_training_ft"${j}"
done

# Go in franka and duplicate file .txt from no_tuning
cd ../Data_generation/python/examples/Franka
rm training_dataset_list.txt
cp ./training_dataset_list_no_ft.txt ./training_dataset_list.txt

cd ../../../../Transformer_for_isaac

cp -r Prova_big_training_NO_ft Prova_big_training_ft_FC


