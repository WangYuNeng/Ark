# python edge_detection_main.py --store --ed_img_path data/ed_28x28.npz --bz 512 --downsample 1
dir=weights
tag=edge-detection-exp
mkdir -p $dir
for seed in {0..5}
do
    save_path=$dir/$seed.npz
    python edge_detection_main.py --plot_evolve 5 --dataset silhouettes --mismatched_edge 0.1 --end_time 1.5 \
    --bz 128 --seed $seed --steps 1  --lr 0.1 --mismatched_node --vectorize \
    --save_weight $save_path --tag $tag --wandb
done

for seed in {0..5}
do
    # Test the model
    # seed+=4
    python edge_detection_main.py --dataset silhouettes --mismatched_edge 0.1 --end_time 1.5 \
    --bz 128 --seed $((seed+4)) --steps 2 --mismatched_node --vectorize \
    --load_weight $save_path --tag $tag --test --wandb
done

# No vectorization
for seed in {0..0}
do
    save_path=$dir/$seed.npz
    python edge_detection_main.py --plot_evolve 5 --dataset silhouettes --mismatched_edge 0.1 --end_time 1.5 \
    --bz 128 --seed $seed --steps 1  --lr 0.1 --mismatched_node  \
    --tag $tag --wandb
done