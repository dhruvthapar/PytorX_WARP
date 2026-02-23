#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"


# path to benchmark
benchmark="$PYTORX_HOME/benchmark/mnist.py"
# benchmark="$PYTORX_HOME/benchmark/biresh-resnet14.py"
# benchmark="$PYTORX_HOME/benchmark/biresh-vgg11.py"
############### Neural network ############################
epochs=40
batch_size=256
test_batch_size=100
crxb_size=128
vdd=3.3
gwire=0.375
gload=0.25
gmax=0.000333
gmin=0.000000333
freq=10e6
test=1

{
python3 $benchmark  --epochs $epochs \
                    --batch_size $batch_size\
                    --test_batch_size $test_batch_size\
                    --crxb_size $crxb_size\
                    --vdd $vdd\
                    --freq $freq\
                    --gwire $gwire\
                    --gload $gload\
                    --gmax $gmax\
                    --gmin $gmin
} &

