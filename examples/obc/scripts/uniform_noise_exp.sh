# Sweep parameters to test the pattern_recog_digit.py example
diff_fn="periodic_mse"
optimizer="adam"
steps=64
bz=1024
lr=1e-1
trans_noise_std=0.001
n_class=5
tag=uniform-noise-exp
wandb="" # --wandb
dir=weights/$tag
mkdir -p $dir
for weight_bits in 1 2 3
do
    for tl in "--trainable_locking" ""
    do
        for tc in "--trainable_coupling" ""
        do
            for weight_init in hebbian
            do
                for seed in {0..4}
                do
                    run_name=bit$weight_bits-$weight_init-seed$seed$tl$tc
                    save_path=$dir/$run_name.npz
                    python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --uniform_noise --vectorize \
                    --trans_noise_std $trans_noise_std --steps $steps --bz $bz --lr $lr --optimizer $optimizer --seed $seed  $wandb --tag $tag \
                    --weight_bits $weight_bits --gumbel_temp_start 10.0 --gumbel_temp_end 1.0 --gumbel_schedule exp  \
                    $tc $tl --pattern_shape 10x6 --save_weight $save_path --weight_init $weight_init --run_name $run_name --no_noiseless
                    python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --uniform_noise --vectorize \
                    --trans_noise_std $trans_noise_std --steps $steps --bz $bz --optimizer $optimizer --seed $((seed+444)) $wandb --tag $tag \
                    --weight_bits $weight_bits $tc $tl --pattern_shape 10x6 --load_weight $save_path --weight_init $weight_init \
                    --run_name $run_name-test --test --no_noiseless
                done
            done
        done
        # Test the loss w/ only hebbian rule
        run_name=bit$weight_bits-$weight_init-seed$seed-no-opt
        python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --uniform_noise --vectorize \
        --trans_noise_std $trans_noise_std --steps $steps --bz $bz --optimizer $optimizer --seed $((seed+444)) $wandb --tag $tag-baseline \
        --weight_bits $weight_bits $tc $tl --pattern_shape 10x6 --weight_init hebbian \
        --run_name $run_name-test --test --no_noiseless
    done
done

# No vectorization

for weight_bits in 1 2 3
do
    for tl in "--trainable_locking" ""
    do
        for tc in "--trainable_coupling" ""
        do
            for weight_init in hebbian
            do
                for seed in {0..0}
                do
                    run_name=bit$weight_bits-$weight_init-seed$seed$tl$tc-novect
                    save_path=$dir/$run_name.npz
                    python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --uniform_noise  \
                    --trans_noise_std $trans_noise_std --steps $steps --bz $bz --lr $lr --optimizer $optimizer --seed $seed  $wandb --tag $tag \
                    --weight_bits $weight_bits --gumbel_temp_start 10.0 --gumbel_temp_end 1.0 --gumbel_schedule exp  \
                    $tc $tl --pattern_shape 10x6 --weight_init $weight_init --run_name $run_name --no_noiseless
                done
            done
        done
    done
done
