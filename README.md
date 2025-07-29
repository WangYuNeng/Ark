# ARK: A Programming Language for Agile Development of Unconventional Computing Paradigms

This package implements the language described in [Design of Novel Analog Compute Paradigms with Ark](https://arxiv.org/abs/2309.08774).

## Build from source

To setup the environment, please have `graphviz`, and a SMT solver in the system and run

```bash
conda create -n ark python=3.11
conda activate ark
pip install -r requirement_torch.txt
pip install -r requirement_benchmark.txt # if you want to run the benchmarking experiments
pip install -r requirement.txt
pip install -e .
```

The order is important, as `torch` depends on an older version of `cudnn` and `jax` depends on a newer version. Install `torch` first and ignore the error message about `cudnn` when installing `jax` as `cudnn` is usually backward compatible.

If you are on macOS and use `brew` to install `z3`, it is possible that z3 solver failed to find the `libz3.dylib` even when z3 is installed with brew. This because somehow z3 looks for librariesin `/op/how/bin/` instead of `/opt/homebrew/lib`. You can get around by running:

```bash
cp /opt/homebrew/lib/libz3.dylib /Users/{username}/miniconda3/envs/ark/lib
```

## RUNME Example

Let's start with an example in oscillator-based computing. Run the following command to optimize oscillator-based pattern recognizer with uniform noise injected to the images.

```bash
cd examples/obc
python3 pattern_recog_main.py --n_class 4 --diff_fn periodic_mse --uniform_noise \
    --vectorize --trans_noise_std 0.01 --steps 64 --bz 1024 --lr 0.1 --optimizer adam \
    --seed 444 --weight_bits 1 --gumbel_temp_start 10.0 --gumbel_temp_end 1.0 --gumbel_schedule exp \
    --trainable_locking --trainable_coupling --pattern_shape 10x6 --weight_init hebbian --no_noiseless --plot_evol 4 --num_plot 4
```

The program should first show plots of example obc evolutions without transient noise and a second plot with transient noise. The program will then run the optimization. Afterward, there will be a final plot showing the evolution of obc with the optimized weights.

For the usage of the arguments, you can refer to `python3 pattern_recog_main.py -h`

A complete experiment for OBC with uniform noise can be run with

```bash
bash scripts/uniform_noise_exp.sh
```

The script enumerates different configurations including random seed, number of DAC bits, choices of trainable parameters, and coupling weight initialization. The results will be logged to the screen.
If [weight-and-bias](https://wandb.ai/site/) is set up, you can log the training and testing runs by modifying line 10 of the script to `wandb="--wandb"`

## More Examples

More examples inside the `examples` directory.

- `obc`: A specification for the oscillator-based computing and a pattern recognition application.
  - `scripts/uniform_noise_exp.sh`: A script to run the OBC pattern recognition optimization under uniform noise.
  - `scripts/snp_noise_exp.sh`: A script to run the OBC edge detection optimization under salt-and-pepper noise.
- `cnn`: A specification for the cellular nonlinear network and an edge-detection application.
  - `scripts/edge_detection.sh`: A script to run the CNN edge detection optimization under random mismatch.
- `nacs_as_nn`: Specifications for Novel Analog Computing Systems as Neural Networks, the systems including CNN, OBC, and CNN with tanh activation.
  - `scripts/classifier_comparison`: A script to run the mnist and fashion-mnist classification tasks with different architectures.
- `tln`: A specification for the transmission-line-network and examaples demonstrating how mismatches affects the system response.
