# python edge_detection_main.py --store --ed_img_path data/ed_28x28.npz --bz 512 --downsample 1
dir=weights
tag=edge-detection-exp
mmstd=0.02
mkdir -p $dir
for seed in {0..5}
do
    save_path=$dir/$seed.npz
    python edge_detection_main.py --plot_evolve 5 --dataset silhouettes --mismatched_edge $mmstd --end_time 1.5 \
    --bz 128 --seed $seed --steps 1  --lr 0.1 --mismatched_node $mmstd --vectorize \
    --save_weight $save_path --tag $tag --wandb
done

for seed in {0..5}
do
    # Test the model
    # seed+=4
    save_path=$dir/$seed.npz
    python edge_detection_main.py --dataset silhouettes --mismatched_edge $mmstd --end_time 1.5 \
    --bz 128 --seed $((seed+4)) --steps 1 --mismatched_node $mmstd --vectorize \
    --load_weight $save_path --tag $tag --test --wandb
done

# Test with the original weight
python edge_detection_main.py --dataset silhouettes --mismatched_edge $mmstd --end_time 1.5 \
    --bz 128 --seed 6 --steps 1 --mismatched_node $mmstd --vectorize \
    --tag $tag --test --wandb


# No vectorization
for seed in {0..0}
do
    python edge_detection_main.py --plot_evolve 5 --dataset silhouettes --mismatched_edge $mmstd --end_time 1.5 \
    --bz 128 --seed $seed --steps 1  --lr 0.1 --mismatched_node $mmstd \
    --tag $tag --wandb
done
