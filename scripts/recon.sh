#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DS_SKIP_CUDA_CHECK=1

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

subj=1
guidance_scale=300000
guidance_strength=2
guidance_scale_list=(300000 30000 100000)
guidance_strength_list=(0.2 0.1 0.4)

# run_cmd="python ../src/recon.py --subject ${subj} --guidance_scale ${guidance_scale} --guidance_strength ${guidance_strength}"
# echo ${run_cmd}
# eval ${run_cmd} 
# set +x
    
# run_cmd="python ../src/recon_no_z.py --subject ${subj} --guidance_scale ${guidance_scale} --guidance_strength ${guidance_strength}"
# echo ${run_cmd}
# eval ${run_cmd} 
# set +x

# for guidance_scale in ${guidance_scale_list[@]}; do
#     for guidance_strength in ${guidance_strength_list[@]}; do
#         run_cmd="python ../src/recon_no_z.py --subject ${subj} --guidance_scale ${guidance_scale} --guidance_strength ${guidance_strength}"
#         echo ${run_cmd}
#         eval ${run_cmd}
#         # Check if the command was successful
#         if [ $? -ne 0 ]; then
#             echo "Error: Command failed with exit code $?"
#             exit 1
#         fi
#     done
# done

# Phase strengths and schedules to sweep (no normal guidance_strength)
schedule_types=(brain_aware fixed linear exponential)
early_list=(0.4 0.6)
mid_list=(0.2 0.3)
late_list=(0.05 0.1)

for guidance_scale in ${guidance_scale_list[@]}; do
    for schedule_type in ${schedule_types[@]}; do
        for early in ${early_list[@]}; do
            for mid in ${mid_list[@]}; do
                for late in ${late_list[@]}; do
                    echo "  Running adaptive guidance (scale: $guidance_scale, schedule: $schedule_type, E:$early M:$mid L:$late)"
                    run_cmd="python ../src/recon_adaptive_guidance.py --subject ${subj} --guidance_scale ${guidance_scale} --use_adaptive_scheduling --schedule_type ${schedule_type} --early_guidance_strength ${early} --mid_guidance_strength ${mid} --late_guidance_strength ${late}"
                    echo "  Command: ${run_cmd}"
                    eval ${run_cmd}
                done
            done
        done
    done
done