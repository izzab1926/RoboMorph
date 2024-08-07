#! /bin/sh

# --------------------------------  TRAIN EXAMPLE ----------------------------------------------------------------

# for batches in 8; do
#     for embd in 192; do
#         for head in 4; do
#             for layers in 12; do
#                 for loss in 'MSE'; do                                                                                                                                    
#                     python train_sim_multiple_seeds.py --train-folder "train" --model-folder "prova_norm" --bias --log-wandb False --loss-function "${loss}" --n-layer "${layers}" --n-head "${head}" --n-embd "${embd}" --batch-size "${batches}" --manuel_pc True   
#                 done
#             done
#         done
#     done
# done

# --------------------------------  TEST EXAMPLE ----------------------------------------------------------------

for out in 'out_ds_big_norm_5'; do  
    for test in 'train'; do
        for num in 800; do
            # Metrics with merge
            python TEST_multiple_model_MERGED.py --test-folder "${test}" --model-folder "${out}" --figure-folder "fig_${out}_${test}_different" --num-test "${num}" --plot-predictions-and-metrics False
            # Metrics without merge
            # python TEST_multiple_model.py --test-folder "${test}" --model-folder "${out}" --figure-folder "fig_recap_${out}_${test}" --num-test "${num}" 
        done
    done
done


# for out in 'out_kuka_franka_reduced_60'; do  
#     for test in 'test_mix'; do
#         for num in 1000; do
#             # Metrics with merge
#             # python TEST_multiple_model_MERGED.py --test-folder "${test}" --model-folder "${out}" --figure-folder "fig_${out}_${test}_different" --num-test "${num}" --plot-predictions-and-metrics True
#             # Metrics without merge
#             # python TEST_multiple_model.py --test-folder "${test}" --model-folder "${out}" --figure-folder "fig_recap_${out}_${test}" --num-test "${num}" 
#             # Metrics with merge reduced
#             python TEST_multiple_model_MERGED_reduced.py --test-folder "${test}" --model-folder "${out}" --figure-folder "fig_${out}_${test}_different" --num-test "${num}" --plot-predictions-and-metrics True

#         done
#     done
# done
