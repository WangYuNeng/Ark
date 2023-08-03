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
echo "con mm1"
python examples/con.py --initialize 1 --atol 0.1 --rtol 0.1
echo "con mm2"
python examples/con.py --initialize 1 --offset_rstd 0.1 
echo "con mm3"
python examples/con.py --initialize 1 --offset_rstd 0.1 --atol 0.1 --rtol 0.1
