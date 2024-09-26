#!/bin/bash

# Test Quantized Model
cd Baseline_Model
python test.py --checkpoint ./model/QAT_ResNet_color/q_RN_step_350.pt

printf "\e[34mFin Test\e[0m\n"
