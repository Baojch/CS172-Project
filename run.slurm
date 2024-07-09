#!/bin/bash

#SBATCH -J test
#SBATCH -p CS172
#SBATCH --cpus-per-task=32
#SBATCH -N 1
#SBATCH --time 7-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=%j.out
#SBATCH --error=%j.err

###body 主体的脚本内容
echo "Start time: `date`"   #显示开始时间
echo "SLURM_JOB_ID: $SLURM_JOB_ID"   #显示作业号
echo "SLURM_NNODES: $SLURM_NNODES"   #显示节点数
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"  #显示每节点任务数
echo "SLURM_NTASKS: $SLURM_NTASKS"   #显示总任务数
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"   #显示作业分区

source /public/home/CS172/baojch-cs172/anaconda3/bin/activate base
python HOPE.py --input_file ./datanpy/ --output_file ./checkpoints/fhad/model- --train --val --resume ./checkpoints/fhad/model-hope-2500.pkl --batch_size 45 --model_def HopeNet --gpu --gpu_number 0 1 2 3 --learning_rate 0.002 --lr_step 60 --lr_step_gamma 0.9 --log_batch 40 --val_epoch 100 --snapshot_epoch 200 --num_iterations 5000 