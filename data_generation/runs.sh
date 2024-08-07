#!/bin/sh


# PHYSICAL
# 1300 steps
# for i in $(seq 2); do
#     for type_of_task in 'FC'; do # 'FS' 'FC'
#         for bounds in 10; do        
#             python generation_franka.py --plot-diagrams True --max-iteration 1300 --headless-mode False --random-initial-positions False --num-envs 32 --osc-task True --type-of-task "${type_of_task}" --control-imposed-file False --control-imposed False --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --save-tensors True                                                                                                                        
#         done
#     done
# done

# for i in $(seq 1); do
#     for frequency in 0.1 0.15 0.2; do
#         for bounds in 10 15 5; do 
#             for type in 'multi_sinusoidal'; do                                                                                                                            
#                 python generation_franka.py --headless-mode False --random-stiffness-dofs True --random-initial-positions True --num-envs 128 --plot-diagrams False --max-iteration 1000 --frequency "${frequency}" --control-imposed-file False --control-imposed True --osc-task False --type-of-input "${type}" --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --save-tensors True --type-of-dataset 'test'
#             done 
#         done
#     done
# done

# for i in $(seq 5); do
#     for type_of_task in 'FC'; do # 'FS' 'FC'
#         for bounds in 10; do        
#             python generation_franka.py --random-stiffness-dofs True --num-envs 2048 --osc-task True --type-of-task "${type_of_task}" --control-imposed-file False --control-imposed False --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'train'
#         done
#     done
# done

# TEST D
# for i in $(seq 2); do
#     for type_of_task in 'FC'; do # 'FS' 'FC'
#         for bounds in 10; do        
#             # python generation_franka.py --num-envs 4096 --osc-task True --type-of-task "${type_of_task}" --control-imposed-file False --control-imposed False --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#             # python generation_franka.py --random-stiffness-dofs True --num-envs 4096 --osc-task True --type-of-task "${type_of_task}" --control-imposed-file False --control-imposed False --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#         done
#     done
# done

# for i in $(seq 1); do
#     for frequency in 0.15; do
#         for bounds in 1 5 8 10; do 
#             for type in 'multi_sinusoidal' 'chirp'; do                                                                                                                            
#                 python generation_franka.py --frequency "${frequency}" --num-envs 3072 --control-imposed-file False --control-imposed True --osc-task False --type-of-input "${type}" --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --save-tensors True --type-of-dataset 'test'
#             done 
#         done
#     done
# done

# TEST D
# for i in $(seq 2); do
#     for type_of_task in 'VS'; do # 'FS' 'FC'
#         for bounds in 5 10; do                                                                                                                                
#             python generation_franka.py --num-envs 4096 --osc-task True --type-of-task "${type_of_task}" --control-imposed-file False --control-imposed False --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#             python generation_franka.py --random-stiffness-dofs True --num-envs 4096 --osc-task True --type-of-task "${type_of_task}" --control-imposed-file False --control-imposed False --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'

#         done
#     done
# done

# for i in $(seq 1); do
#     for type_of_task in 'FS'; do # 'FS' 'FC'
#         for bounds in 10; do                                                                                                                                
#             python generation_franka.py --num-envs 4096 --osc-task True --type-of-task "${type_of_task}" --control-imposed-file False --control-imposed False --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#         done
#     done
# done

# for i in $(seq 1 1); do
#     done
# done

# for i in $(seq 1); do
#     for type_of_task in 'FS'; do # 'FS' 'FC'
#         for bounds in 10; do                                                                                                                                
#             python generation_franka.py --num-envs 4096 --osc-task True --type-of-task "${type_of_task}" --control-imposed-file False --control-imposed False --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#         done
#     done
# done

# for i in $(seq 1 1); do
#     for type_of_task in 'VS' 'FS' 'FC'; do # 
#         for bounds in 10; do                                                                                                                                
#             python generation_franka.py --headless-mode False --num-envs 32 --osc-task True --type-of-task "${type_of_task}" --control-imposed-file False --control-imposed False --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" 
#         done
#     done
# done

for i in $(seq 10); do
    for frequency in 0.15; do
        for bounds in 10; do 
            for type in 'multi_sinusoidal'; do                                                                                                                            
                python generation_franka.py --frequency "${frequency}" --num-envs 1024 --control-imposed-file False --control-imposed True --osc-task False --type-of-input "${type}" --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --save-tensors True --type-of-dataset 'train'
            done 
        done
    done
done

# python generation_franka.py --seed 392 --num-envs 4096 --frequency 0.1 --control-imposed-file False --control-imposed True --osc-task False --type-of-input 'chirp' --lower-bound-mass 10 --higher-bound-mass 10 --type-of-dataset 'test'

# Test_B --> (Dataset3 type) Task function but with in/out distribution
# Need to be split in 2! -->  MS and chirp

# for i in $(seq 1); do
#     for frequency in 0.15; do
#         for bounds in 1 5 8 10; do 
#             for type in 'multi_sinusoidal' 'chirp'; do                                                                                                                            
#                 python generation_franka.py --frequency "${frequency}" --num-envs 3072 --control-imposed-file False --control-imposed True --osc-task False --type-of-input "${type}" --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --save-tensors True --type-of-dataset 'test'
#             done 
#         done
#     done
# done

# for i in $(seq 6); do
#     for type_of_task in 'VS'; do
#        for bounds in 10; do      
#             python generation_franka.py --type-of-task "${type_of_task}" --random-stiffness-dofs False --max-iteration 1200 --control-imposed-file False --control-imposed False --osc-task True --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --save-tensors False                                                                                        
#        done
#    done
# done


# TEST KUKA

# for i in $(seq 1); do
#     for frequency in 0.1 0.15 0.2; do
#         for bounds in 5 10; do                                                                                                                                
#             # python generation_franka.py --plot-diagrams True --headless-mode False --num-envs 32 --random-stiffness-dofs True --frequency "${frequency}" --control-imposed-file False --control-imposed True --osc-task False --type-of-input 'chirp' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --save-tensors False 
#             python generation_kuka.py --plot-diagrams False --num-envs 3072 --random-stiffness-dofs True --frequency "${frequency}" --control-imposed-file False --control-imposed True --osc-task False --type-of-input 'multi_sinusoidal' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#         done
#     done
# done


# ----------------- INSTRUCTIONS ----------------------

# Torque From Function

# python generation_franka.py --plot-diagrams True --disable-gravity False --headless-mode True --num-envs 32 --control-imposed-file False --control-imposed True --osc-task False  --lower-bound-mass 10 --higher-bound-mass 10 --save-tensors False 

# # Torque From OSC task
# python generation_franka.py --plot-diagrams True --disable-gravity False --headless-mode False --num-envs 32 --control-imposed-file False --control-imposed False --osc-task True  --lower-bound-mass 10 --higher-bound-mass 10 --save-tensors True --type-of-dataset 'train'

# Torque From file -- It must be specified inside the file the name!
# It need to be 0% error!
# compensate must be set False, if already included in the file.
# python generation_franka.py --plot-diagrams True --disable-gravity False --headless-mode False --num-envs 32 --control-imposed-file True --control-imposed True --osc-task False  --lower-bound-mass 10 --higher-bound-mass 10 --save-tensors True --type-of-dataset 'train' 
