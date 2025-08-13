# Sweep parameters to test the pattern_recog_digit.py example
diff_fn="periodic_mse"
optimizer="adam"
steps=64
bz=1024
lr=1e-1
trans_noise_std=0.001
n_class=5
tag=blackbox-exp
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
                    $tc $tl --pattern_shape 10x6 --save_weight $save_path --weight_init $weight_init --run_name $run_name --no_noiseless \
                    --blackbox_opt ax
                    python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --uniform_noise --vectorize \
                    --trans_noise_std $trans_noise_std --steps $steps --bz $bz --optimizer $optimizer --seed $((seed+444)) $wandb --tag $tag \
                    $tc $tl --pattern_shape 10x6 --load_weight $save_path --weight_init $weight_init \
                    --run_name $run_name-test --test --no_noiseless
                done
            done
        done
    done
done


diff_fn="periodic_mse"
optimizer="adam"
steps=64
test_steps=16
bz=1024
lr=1e-2
snp_prob=0.1
n_class=5
l1_norm_weight=1e-4
trans_noise_std=0.001
weight_init=hebbian
locking_strength=2.0
for connection in "all" "neighbor" 
do
    dir=weights/$tag
    mkdir -p $dir
    if [[ $connection == "neighbor" ]]; then
        l1_norm_weight=0.0
    fi
    for fcw in ""
    do
        for seed in {0..4}
        do
            run_name=seed$seed-conn-$connection$fcw
            save_path=$dir/$run_name.npz
            python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --vectorize --connection $connection \
            --trans_noise_std $trans_noise_std --steps $steps --bz $bz --lr $lr --optimizer $optimizer --seed $seed   --tag $tag \
            --pattern_shape 10x6 --save_weight $save_path --weight_init $weight_init --run_name $run_name --no_noiseless \
            --trainable_locking --trainable_coupling $fcw --l1_norm_weight $l1_norm_weight --snp_prob $snp_prob \
            --locking_strength $locking_strength --blackbox_opt ax $wandb
            # Test the model
            python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --vectorize --connection $connection --test \
            --trans_noise_std $trans_noise_std --steps $test_steps --bz $bz  --seed $((seed+444))  --wandb --tag $tag \
            --pattern_shape 10x6 --load_weight $save_path --weight_init $weight_init --run_name $run_name-test --no_noiseless \
            --trainable_locking --trainable_coupling $fcw --l1_norm_weight $l1_norm_weight --snp_prob $snp_prob \
            --locking_strength $locking_strength
        done
    done
done
