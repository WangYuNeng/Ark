# python edge_detection_main.py --store --ed_img_path data/ed_28x28.npz --bz 512 --downsample 1
dir=weights
tag=blackbox-edge-detection
mkdir -p $dir
for seed in {0..5}
do
    save_path=$dir/blackbox$seed.npz
    python edge_detection_main.py --plot_evolve 5 --dataset silhouettes --mismatched_edge 0.1 --end_time 1.5 \
    --bz 128 --seed $seed --steps 2  --mismatched_node --vectorize \
    --save_weight $save_path --tag $tag --wandb --blackbox_opt ax

    python edge_detection_main.py --dataset silhouettes --mismatched_edge 0.1 --end_time 1.5 \
    --bz 128 --seed $((seed+444)) --steps 2 --mismatched_node --vectorize \
    --load_weight $save_path --tag $tag --test --wandb
done

for seed in {0..5}
do
    save_path=$dir/blackbox-lim$seed.npz
    python edge_detection_main.py --plot_evolve 5 --dataset silhouettes --mismatched_edge 0.1 --end_time 1.5 \
    --bz 128 --seed $seed --steps 2  --mismatched_node --vectorize \
    --save_weight $save_path --tag $tag --wandb --blackbox_opt ax --limited_range

    python edge_detection_main.py --dataset silhouettes --mismatched_edge 0.1 --end_time 1.5 \
    --bz 128 --seed $((seed+444)) --steps 2 --mismatched_node --vectorize \
    --load_weight $save_path --tag $tag --test --wandb
done

