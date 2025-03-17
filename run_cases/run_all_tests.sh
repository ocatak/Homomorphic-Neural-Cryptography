#!/bin/bash

DROPOUT_RATES=(0.01 0.1 0.2 0.3 0.4 0.5)
CURVES=(secp256k1 secp256r1)
BATCH_SIZE=448
EPOCHS=100

for CURVE in "${CURVES[@]}"; do
  python key/EllipticCurve.py -batch $BATCH_SIZE -curve $CURVE
done

for RATE in "${DROPOUT_RATES[@]}"; do
  for CURVE in "${CURVES[@]}"; do
    echo "Training with dropout rate: $RATE, curve: $CURVE"
    python ../training.py -rate $RATE -epoch $EPOCHS -batch $BATCH_SIZE -curve $CURVE
  done
done

python ../results.py