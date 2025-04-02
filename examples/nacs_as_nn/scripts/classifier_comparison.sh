tag=mismatch
input_type=initial_state
batch_size=256
n_epochs=100
early_stopping=4
trainable_init=uniform
lr=0.001
hidden_size=512
neighbor_dist=1

save_dir=saved/$tag
# Create the save directory if it doesn't exist
mkdir -p $save_dir

for seed in {0..3}
do
    for sys in CNN CANN None
    do
        for downsample in 1 2 4
        do
            for mismatch_rstd in 0.1
            do
                for dataset in mnist fashion_mnist
                do
                    name=${dataset}_ds${downsample}_mismatch${mismatch_rstd}_seed${seed}
                    save_file=$save_dir/$name.eqx
                    python train.py --tag $tag --seed $seed --lr $lr --neighbor_dist $neighbor_dist --sys $sys \
                    --image_downsample $downsample --dataset $dataset --n_epochs $n_epochs --hidden_size $hidden_size --input_type $input_type \
                    --testing --batch_size $batch_size --early_stopping $early_stopping --trainable_init $trainable_init \
                    --mismatch_rstd $mismatch_rstd --save_path $save_file --run_name $name --wandb
                done
            done
        done
    done
done

