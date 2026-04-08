#!/bin/bash

# run other models
python3 multimodal_trainer.py --run_baseline --loss_func tversky --epochs 300 --batch_size 6 --grad_accumulation 2 --mixed_precision

# run proposed model
python3 multimodal_trainer.py --loss_func tversky --epochs 300 --batch_size 6 --grad_accumulation 2 --mixed_precision