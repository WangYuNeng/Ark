# python edge_detection_main.py --store --ed_img_path data/ed_28x28.npz --bz 512 --downsample 1
dir=weights
tag=blackbox-edge-detection
mmstd=0.04
mkdir -p $dir
for seed in {0..4}
do
    save_path=$dir/blackbox$seed.npz
    python edge_detection_main.py --plot_evolve 5 --dataset silhouettes --mismatched_edge $mmstd --end_time 1.0 \
    --bz 128 --seed $seed --steps 1  --mismatched_node $mmstd --vectorize \
    --save_weight $save_path --tag $tag --wandb --blackbox_opt ax

    python edge_detection_main.py --dataset silhouettes --mismatched_edge $mmstd --end_time 1.0 \
    --bz 128 --seed $((seed+444)) --steps 2 --mismatched_node $mmstd --vectorize \
    --load_weight $save_path --tag $tag --test --wandb
done

for seed in {0..4}
do
    save_path=$dir/blackbox-lim$seed.npz
    python edge_detection_main.py --plot_evolve 5 --dataset silhouettes --mismatched_edge $mmstd --end_time 1.0 \
    --bz 128 --seed $seed --steps 1  --mismatched_node $mmstd --vectorize \
    --save_weight $save_path --tag $tag --wandb --blackbox_opt ax --limited_range

    python edge_detection_main.py --dataset silhouettes --mismatched_edge $mmstd --end_time 1.0 \
    --bz 128 --seed $((seed+444)) --steps 2 --mismatched_node $mmstd --vectorize \
    --load_weight $save_path --tag $tag --test --wandb
done

