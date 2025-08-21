# !/bin/bash
dir=weights
tag=blackbox-tln
mkdir -p $dir
for seed in {0..10};
do
    readout_time="10e-9"
    n_time_point=100
    chl_per_bit=32
    lr="5e-3"
    save_path=$dir/blackbox$seed.npz
    python3 train_w_optcompile.py --seed $seed --readout_time $readout_time \
    --inst_per_batch 8 --chl_per_bit $chl_per_bit --steps 24  \
    --n_branch 32 --n_time_point $n_time_point --normalize --wandb --tag $tag --save_weight $save_path \
    --vectorize --blackbox_opt ax

    # Testing
    python3 train_w_optcompile.py --seed $((seed+444)) --readout_time $readout_time \
    --inst_per_batch 8 --chl_per_bit $chl_per_bit --steps 24  \
    --n_branch 32 --n_time_point $n_time_point --normalize --wandb --tag $tag --load_weight $save_path \
    --vectorize --test
done


