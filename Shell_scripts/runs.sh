#!/bin/sh


# ----------------------- February 2024 ----------------------------------------------

# used for dataset_mix2 ( in addition to Dataset_mix1)
# for i in $(seq 1 8); do
#     for frequency in 0.1 0.15 0.2; do
#         for bounds in 15; do                                                                                                                                
#             python scratch_2024b.py --num-envs 4096 --random-stiffness-dofs True --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input 'multi_sinusoidal' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#             python scratch_2024b.py --num-envs 4096 --random-stiffness-dofs True --disable-gravity False --control-imposed-file False --control-imposed False --xy-circle-task True  --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#             python scratch_2024b.py --num-envs 4096 --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input 'multi_sinusoidal' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#             python scratch_2024b.py --num-envs 4096 --disable-gravity False --control-imposed-file False --control-imposed False --xy-circle-task True  --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#         done
#     done
# done

# Dataset_mix1
#for i in $(seq 1 20); do
#    for frequency in 0.1 0.15 0.2; do
#        for bounds in 15; do                                                                                                                                
#            python scratch_2024b.py --num-envs 4096 --random-stiffness-dofs True --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input 'multi_sinusoidal' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'train'
#            python scratch_2024b.py --num-envs 4096 --random-stiffness-dofs True --disable-gravity False --control-imposed-file False --control-imposed False --xy-circle-task True  --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'train'
#            python scratch_2024b.py --num-envs 4096 --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input 'multi_sinusoidal' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'train'
#            python scratch_2024b.py --num-envs 4096 --disable-gravity False --control-imposed-file False --control-imposed False --xy-circle-task True  --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'train'
#        done
#    done
#done

# Dataset_2
#for i in $(seq 1 115); do
#    for frequency in 0.15; do
#        for bounds in 10; do                                                                                                                                
#            python scratch_2024b.py --num-envs 4096 --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input 'multi_sinusoidal' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'train' 
#        done
#    done
#done

# Dataset_3
#for i in $(seq 1 40); do
#    for frequency in 0.1 0.15 0.2; do
#        for bounds in 10; do   
#            for type_input in 'chirp' 'multi_sinusoidal'; do                                                                                                                               
#                python scratch_2024b.py --num-envs 4096 --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input "${type_input}" --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'train' 
#            done
#        done
#    done
#done

# -----------------------------------------------------------------------

#Test_A --> (Dataset2 type) -- Task function but with in/out distribution
#for i in $(seq 1); do
#    for frequency in 0.1 0.15 0.2; do
#        for bounds in 5 10 15; do                                                                                                                                
#            python scratch_2024b.py --num-envs 4096 --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input 'multi_sinusoidal' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test' 
#        done
#    done
#done

#Test_B --> (Dataset3 type) Task function but with in/out distribution
# Need to be split in 2! -->  MS and chirp
#for i in $(seq 1); do
#    for frequency in 0.1 0.15 0.2; do
#        for bounds in 5 10 15; do   
#            for type_input in 'chirp' 'multi_sinusoidal'; do                                                                                                                               
#                python scratch_2024b.py --num-envs 4096 --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input "${type_input}" --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test' 
#            done
#        done
#    done
#done

# Test_C --> (sort of Dataset_mix1 type [not exactly the same]) OSC & Torque function. bounds inside! consider stiffness here changes!!!
# for i in $(seq 1 15); do
#     for frequency in 0.1 0.15 0.2; do
#         for bounds in 15; do                                                                                                                                
#             python scratch_2024b.py --num-envs 4096 --random-stiffness-dofs True --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input 'multi_sinusoidal' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#             python scratch_2024b.py --num-envs 4096 --random-stiffness-dofs True --disable-gravity False --control-imposed-file False --control-imposed False --xy-circle-task True  --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#             #python scratch_2024b.py --num-envs 4096 --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input 'multi_sinusoidal' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#             #python scratch_2024b.py --num-envs 4096 --disable-gravity False --control-imposed-file False --control-imposed False --xy-circle-task True  --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#         done
#     done
# done

# Test_C --> (sort of Dataset_mix1 type [not exactly the same]) OSC & Torque function. bounds inside! consider stiffness here changes!!!

# for i in $(seq 1 18); do
#     for frequency in 0.15; do
#         for bounds in 10; do                                                                                                                                
#             python scratch_2024b.py --num-envs 4096 --random-stiffness-dofs True --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input 'multi_sinusoidal' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'train'
#         done
#     done
# done

#for i in $(seq 1 1); do
#    for frequency in 0.1 0.15 0.2 0.25; do
#        for bounds in 10; do                                                                                                                                
#            python scratch_2024b.py --num-envs 128 --plot-diagrams True --headless-mode False --random-stiffness-dofs False --disable-gravity False --control-imposed-file False --control-imposed True --type-of-input 'combination' --xy-circle-task False  --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'train'
#        done
#    done
#done


#Test_B --> (Dataset3 type) Task function but with in/out distribution
# Need to be split in 2! -->  MS and chirp
for i in $(seq 1); do
    for frequency in 0.05 0.15 0.3; do
        for bounds in 5 10 15; do   
            for type_input in 'chirp' 'multi_sinusoidal'; do                                                                                                                               
                python scratch_2024b.py --num-envs 4096 --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input "${type_input}" --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test' 
            done
        done
    done
done

# -----------------------------------------------------------------------


#for i in $(seq 1 30); do
#    for frequency in 0.1 0.15 0.2; do
#        for bounds in 15; do                                                                                                                                
#            # python scratch_2024b.py --num-envs 4096 --random-stiffness-dofs True --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input 'multi_sinusoidal' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#            # python scratch_2024b.py --num-envs 4096 --random-stiffness-dofs True --disable-gravity False --control-imposed-file False --control-imposed False --xy-circle-task True  --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'test'
#            python scratch_2024b.py --num-envs 4096 --frequency "${frequency}" --control-imposed-file False --control-imposed True --xy-circle-task False --type-of-input 'multi_sinusoidal' --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'train'
#            python scratch_2024b.py --num-envs 4096 --disable-gravity False --control-imposed-file False --control-imposed False --xy-circle-task True  --lower-bound-mass "${bounds}" --higher-bound-mass "${bounds}" --type-of-dataset 'train'
#        done
#    done
#done

# python scratch_2024b.py --plot-diagrams True --disable-gravity False --random-stiffness-dofs True --headless-mode False --num-envs 32 --control-imposed-file False --control-imposed True --xy-circle-task False  --lower-bound-mass 10 --higher-bound-mass 10 --save-tensors True --type-of-dataset 'train'

# From Function
# python scratch_2024b.py --plot-diagrams True --disable-gravity False --headless-mode False --num-envs 32 --control-imposed-file False --control-imposed True --xy-circle-task False  --lower-bound-mass 10 --higher-bound-mass 10 --save-tensors True --type-of-dataset 'train'

# # From OSC task
# python scratch_2024b.py --plot-diagrams True --disable-gravity False --headless-mode False --num-envs 32 --control-imposed-file False --control-imposed False --xy-circle-task True  --lower-bound-mass 10 --higher-bound-mass 10 --save-tensors True --type-of-dataset 'train'


# From file -- It must be specified inside the file the name!
# It need to be 0% error!.
# compensate must be set False, if already included in the file.
# python scratch_2024b.py --plot-diagrams True --disable-gravity False --headless-mode False --num-envs 32 --control-imposed-file True --control-imposed True --xy-circle-task False  --lower-bound-mass 10 --higher-bound-mass 10 --save-tensors True --type-of-dataset 'train' 
