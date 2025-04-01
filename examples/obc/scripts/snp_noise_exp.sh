# Sweep parameters to test the pattern_recog_digit.py example
diff_fn="periodic_mse"
optimizer="adam"
steps=64
bz=1024
lr=1e-1
snp_prob=0.1
n_class=5
tag=snp-noise-exp
l1_norm_weight=1e-3
trans_noise_std=0.01
weight_init=hebbian
for seed in {0..3}
do
    dir=weights/$tag
    mkdir -p $dir
    for connection in "neighbor" "all"
    do
        for fcw in "--fix_coupling_weight" ""
        do
            run_name=seed$seed-conn-$connection$fcw
            save_path=$dir/$run_name.npz
            python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --vectorize --connection $connection \
            --trans_noise_std $trans_noise_std --steps $steps --bz $bz --lr $lr --optimizer $optimizer --seed $seed  --wandb --tag $tag \
            --pattern_shape 10x6 --save_weight $save_path --weight_init $weight_init --run_name $run_name --no_noiseless \
            --trainable_locking --trainable_coupling $fcw --l1_norm_weight $l1_norm_weight --snp_prob $snp_prob
            
            # Test the model
            python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --vectorize --connection $connection --test \
            --trans_noise_std $trans_noise_std --steps $steps --bz $bz  --seed $seed  --wandb --tag $tag \
            --pattern_shape 10x6 --load_weight $save_path --weight_init $weight_init --run_name $run_name-test --no_noiseless \
            --trainable_locking --trainable_coupling $fcw --l1_norm_weight $l1_norm_weight --snp_prob $snp_prob

            # Test the model w/ droping 50% of the weights
            # Do only when "all" and not fix_coupling_weight
            if [[ $connection == "all" && $fcw == "" ]]; then
                python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --vectorize --connection $connection --test \
                --trans_noise_std $trans_noise_std --steps $steps --bz $bz --seed $seed  --wandb --tag $tag \
                --pattern_shape 10x6 --load_weight $save_path --weight_init $weight_init --run_name $run_name-test-wd --no_noiseless \
                --trainable_locking --trainable_coupling $fcw --l1_norm_weight $l1_norm_weight --snp_prob $snp_prob --weight_drop_ratio 0.5
            fi
        done
    done
done
