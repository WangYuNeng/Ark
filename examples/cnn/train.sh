# python edge_detection_main.py --store --ed_img_path data/ed_28x28.npz --bz 512 --downsample 1
dir=weights
mkdir -p $dir
for seed in {0..9}
do
    python edge_detection_main.py --plot_evolve 5 --dataset silhouettes --mismatched_edge 0.1 --end_time 1.5 \
    --wandb --bz 128 --seed $seed --steps 1  --lr 0.1 --mismatched_node \
    --save_weight $dir/$seed --tag train-cnn-base
done