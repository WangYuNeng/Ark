# Generate the experiment figures in the paper
echo "Experiment 1: Generate figures in the example and evaluation sections"
echo ""

echo "Generate TLN  Figures (Section 2)"
cd examples/tln
python tln_example.py

echo "Generate CNN Figures (Section 6.1)"
cd ../cnn
python edge_detection.py -p

echo "Generate OBC Figures (Section 6.2)"
cd ../obc
python max_cut.py  --n_cycle 5 -p
python max_cut.py  --n_cycle 5 --offset_rstd 0.01 -p
cd ../..