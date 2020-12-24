#!/bin/bash

# python3 program.py clean_validation_data_dir bd_mode_dir

python3 fine_prune.py noneed "models/sunglasses_bd_net.h5"
python3 fine_prune.py noneed "models/anonymous_1_bd_net.h5"
python3 fine_prune.py noneed "models/anonymous_2_bd_net.h5"
python3 fine_prune.py noneed "models/multi_trigger_multi_target_bd_net.h5"
