tag=nacs-aic # analog-to-information converter
input_type=initial_state
batch_size=256
n_epochs=100
early_stopping=6
trainable_init=uniform
lr=0.001
hidden_size=512
neighbor_dist=1
mismatch_rstd=0.1

save_dir=saved/$tag
# Create the save directory if it doesn't exist
mkdir -p $save_dir

for downsample in 4 2 1
do
    for sys in None CNN
    do
        for nbits in 1 2 8
        do
            for seed in {0..5}
            do
                for dataset in mnist fashion_mnist
                do
                    # skip the case when nbits=8 and downsample!=1 because that nbits=8 is only for
                    # the baselin (full precision) case and don't need to test when downsampled
                    if [ $nbits -eq 8 ] && [ $downsample -ne 1 ]; then
                        continue
                    fi
                    name=${dataset}_ds${downsample}_mismatch${mismatch_rstd}_seed${seed}_bits$nbits
                    save_file=$save_dir/$name.eqx
                    python train.py --tag $tag --seed $seed --lr $lr --neighbor_dist $neighbor_dist --sys $sys \
                    --image_downsample $downsample --dataset $dataset --n_epochs $n_epochs --hidden_size $hidden_size --input_type $input_type \
                    --testing --batch_size $batch_size --early_stopping $early_stopping --trainable_init $trainable_init \
                    --mismatch_rstd $mismatch_rstd --save_path $save_file --run_name $name --wandb --output_quantization_bits $nbits
                done
            done
        done
    done
done

