echo "Experiment 2: Solving max-cut with different OBC dialect and synchronization schemes (Table 1)"
echo ""

cd examples/obc
echo "obc, d=0.01pi"
python max_cut.py  --n_cycle 5 # -p
echo "obc, d=0.1pi"
python max_cut.py  --n_cycle 5 --atol 0.1
echo "ofs-obc, d=0.01pi"
python max_cut.py  --n_cycle 5 --offset_rstd 0.01 # -p
echo "ofs-obc, d=0.1pi"
python max_cut.py  --n_cycle 5 --offset_rstd 0.01 --atol 0.1
cd ../..