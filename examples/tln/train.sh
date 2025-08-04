# !/bin/bash
dir=weights
mkdir -p $dir
for seed in {0..10};
do
    readout_time="10e-9"
    n_time_point=100
    chl_per_bit=32
    lr="5e-3"
    python3 train_w_optcompile.py --seed $seed --readout_time $readout_time \
    --inst_per_batch 8 --chl_per_bit $chl_per_bit --steps 24 --logistic_k 60 --learning_rate $lr \
    --n_branch 32 --n_time_point $n_time_point --normalize --wandb --tag train-tln-base --save_weight $dir/$seed \
    --vectorize
done

# No vectorization
for seed in {0..0};
do
    readout_time="10e-9"
    n_time_point=100
    chl_per_bit=32
    lr="5e-3"
    python3 train_w_optcompile.py --seed $seed --readout_time $readout_time \
    --inst_per_batch 8 --chl_per_bit $chl_per_bit --steps 24 --logistic_k 60 --learning_rate $lr \
    --n_branch 32 --n_time_point $n_time_point --normalize --wandb --tag train-tln-base
done
# If increase the inst_per_batch to 16, GPU won't have enough memory