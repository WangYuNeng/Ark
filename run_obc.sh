echo "Running obc example, compare the correct rate of ideal and mismatched implementation"
echo "obc ideal"
python examples/obc.py  --n_cycle 5 # -p
echo "obc ideal+tol"
python examples/obc.py  --n_cycle 5 --atol 0.1
echo "obc offset"
python examples/obc.py  --n_cycle 5 --offset_rstd 0.01 # -p
echo "obc offset+tol"
python examples/obc.py  --n_cycle 5 --offset_rstd 0.01 --atol 0.1