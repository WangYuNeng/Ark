# Sweep parameters to test the pattern_recog_digit.py example
diff_fn="periodic_mse"
optimizer="adam"
steps=64
bz=1024
lr=1e-1
train_noise_std=0.025
n_class=5
tag=train-obc-base
for seed in {0..9}
do
    for weight_bits in 1 2 3
    do
        dir=weights/nbit$weight_bits
        mkdir -p $dir
        for weight_init in hebbian random
        do
                python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --uniform_noise \
                --trans_noise_std $trans_noise_std --steps $steps --bz $bz --lr $lr --optimizer $optimizer --seed $seed  --wandb --tag $tag \
                --weight_bits $weight_bits --gumbel_temp_start 10.0 --gumbel_temp_end 1.0 --plot 5 --gumbel_schedule exp --num_plot 8 \
                --no_noiseless_train --pattern_shape 10x6 --save_weight $dir/$weight_init-tran-noise$trans_noise_std-seed$seed --weight_init $weight_init

                python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --uniform_noise \
                --trans_noise_std $trans_noise_std --steps $steps --bz $bz --lr $lr --optimizer $optimizer --seed $seed  --wandb --tag $tag \
                --weight_bits $weight_bits --gumbel_temp_start 10.0 --gumbel_temp_end 1.0 --plot 5 --gumbel_schedule exp --num_plot 8 \
                --trainable_coupling --no_noiseless_train --pattern_shape 10x6 --save_weight $dir/trainc-$weight_init-tran-noise$trans_noise_std-seed$seed --weight_init $weight_init

                python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --uniform_noise \
                --trans_noise_std $trans_noise_std --steps $steps --bz $bz --lr $lr --optimizer $optimizer --seed $seed  --wandb --tag $tag \
                --weight_bits $weight_bits --gumbel_temp_start 10.0 --gumbel_temp_end 1.0 --plot 5 --gumbel_schedule exp --num_plot 8 \
                --trainable_locking --no_noiseless_train --pattern_shape 10x6 --save_weight $dir/trainl-$weight_init-tran-noise$trans_noise_std-seed$seed --weight_init $weight_init

                python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --uniform_noise \
                --trans_noise_std $trans_noise_std --steps $steps --bz $bz --lr $lr --optimizer $optimizer --seed $seed  --wandb --tag $tag \
                --weight_bits $weight_bits --gumbel_temp_start 10.0 --gumbel_temp_end 1.0 --plot 5 --gumbel_schedule exp --num_plot 8 \
                --trainable_locking --trainable_coupling --no_noiseless_train --pattern_shape 10x6 --save_weight $dir/trainlc-$weight_init-tran-noise$trans_noise_std-seed$seed --weight_init $weight_init
        done
    done
done
