# Generate the experiment figures in the paper
echo "tln"
cd examples/tln
# python tln_example.py
echo "cnn"
cd ../cnn
# python edge_detection.py -p
cd ../obc
echo "obc ideal"
python max_cut.py  --n_cycle 5 # -p
echo "obc ideal+tol"
python max_cut.py  --n_cycle 5 --atol 0.1
echo "obc offset"
python max_cut.py  --n_cycle 5 --offset_rstd 0.01 # -p
echo "obc offset+tol"
python max_cut.py  --n_cycle 5 --offset_rstd 0.01 --atol 0.1
cd ../..