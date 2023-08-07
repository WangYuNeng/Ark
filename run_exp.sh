# Generate the experiment figures in the paper
# echo "Running tln example, results will be saved in examples/tln.png"
# python examples/tln.py

# echo "Running cnn example, this will take a few minutes. Results will be saved in examples/cnn_images"
# python examples/cnn.py -p

echo "Running con example, compare the correct rate of ideal and mismatched implementation"
echo "cdg metamodel"
python examples/cdg_metamodel.py
echo "tln example"
python examples/tln_example.py
echo "tln"
python examples/tln.py
echo "cnn"
python examples/cnn.py -p
echo "con ideal"
python examples/con.py --initialize 1 --n_cycle 12
echo "con ideal+tol"
python examples/con.py --initialize 1 --n_cycle 12 --atol 0.1
echo "con offset"
python examples/con.py --initialize 1 --n_cycle 12 -offset_rstd 0.1 
echo "con offset+tol"
python examples/con.py --initialize 1 --n_cycle 12 --offset_rstd 0.1 --atol 0.1
echo "con interconnection"
python examples/con_interconnect.py