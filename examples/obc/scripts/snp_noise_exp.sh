# Sweep parameters to test the pattern_recog_digit.py example
diff_fn="periodic_mse"
optimizer="adam"
steps=64
test_steps=16
bz=1024
lr=1e-2
snp_prob=0.1
n_class=5
tag=snp-noise-exp
l1_norm_weight=1e-4
trans_noise_std=0.001
weight_init=hebbian
locking_strength=2.0
dir=weights/$tag
mkdir -p $dir
for connection in "all" "neighbor" 
do
    if [[ $connection == "neighbor" ]]; then
        l1_norm_weight=0.0
    fi
    for fcw in "" "--fix_coupling_weight"
    do
        for seed in {0..4}
        do
            run_name=seed$seed-conn-$connection$fcw
            save_path=$dir/$run_name.npz
            python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --vectorize --connection $connection \
            --trans_noise_std $trans_noise_std --steps $steps --bz $bz --lr $lr --optimizer $optimizer --seed $seed  --wandb --tag $tag \
            --pattern_shape 10x6 --save_weight $save_path --weight_init $weight_init --run_name $run_name --no_noiseless \
            --trainable_locking --trainable_coupling $fcw --l1_norm_weight $l1_norm_weight --snp_prob $snp_prob \
            --locking_strength $locking_strength
        done
    done
done


for connection in "all" "neighbor" 
do
    if [[ $connection == "neighbor" ]]; then
        l1_norm_weight=0.0
    fi
    for fcw in "" "--fix_coupling_weight"
    do
        for seed in {0..4}
        do
            run_name=seed$seed-conn-$connection$fcw
            save_path=$dir/$run_name.npz

            # Test the model
            python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --vectorize --connection $connection --test \
            --trans_noise_std $trans_noise_std --steps $test_steps --bz $bz  --seed $((seed+444))  --wandb --tag $tag \
            --pattern_shape 10x6 --load_weight $save_path --weight_init $weight_init --run_name $run_name-test --no_noiseless \
            --trainable_locking --trainable_coupling $fcw --l1_norm_weight $l1_norm_weight --snp_prob $snp_prob \
            --locking_strength $locking_strength

            # Test the model w/ droping the weights
            # Do only when "all" and not fix_coupling_weight
            if [[ $connection == "all" && $fcw == "" ]]; then
                for weight_drop_ratio in 0.5 0.7 0.9
                do
                    python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn  --vectorize --connection $connection --test \
                    --trans_noise_std $trans_noise_std --steps $test_steps --bz $bz --seed $((seed+444))  --wandb --tag $tag \
                    --pattern_shape 10x6 --load_weight $save_path --weight_init $weight_init --run_name $run_name-test-wd$weight_drop_ratio --no_noiseless \
                    --trainable_locking --trainable_coupling $fcw --l1_norm_weight $l1_norm_weight --snp_prob $snp_prob --weight_drop_ratio $weight_drop_ratio
                done
            fi
        done
    done
done

# No vectorization
for connection in "all" "neighbor" 
do
    dir=weights/$tag
    mkdir -p $dir
    if [[ $connection == "neighbor" ]]; then
        l1_norm_weight=0.0
    fi
    for fcw in ""
    do
        for seed in {0..0}
        do
            run_name=seed$seed-conn-$connection$fcw-novect
            save_path=$dir/$run_name.npz
            python3 pattern_recog_main.py --n_class $n_class --diff_fn $diff_fn --connection $connection \
            --trans_noise_std $trans_noise_std --steps $steps --bz $bz --lr $lr --optimizer $optimizer --seed $seed  --wandb --tag $tag \
            --pattern_shape 10x6  --weight_init $weight_init --run_name $run_name --no_noiseless \
            --trainable_locking --trainable_coupling $fcw --l1_norm_weight $l1_norm_weight --snp_prob $snp_prob \
            --locking_strength $locking_strength
        done
    done
done
