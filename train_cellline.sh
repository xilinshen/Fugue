python main.py \
--file "./data/splatter_simulation.npz" \
--arch densenet21 \
--batch-size 32 \
--dist-url "tcp://localhost:10006" \
--outdir "./cellline_unscale/" \
--mlp --moco-k 448 --moco-m 0.999 \
--in_features 1959 \
--num_batches 3 \
--shuffle-ratio 0.1 \
--randomzero-ratio 0.3 \
--epochs 100 \
--multiprocessing-distributed \
#--load-split-file \
#--split-savedir "./tmp/" \
#--split-now 
