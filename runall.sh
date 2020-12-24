#!/bin/bash

# python3 program.py data_path model_path trigger_label_paht


python3 eval_defense.py data/anonymous_1_poisoned_data.h5 models/anonymous_1_bd_net.h5 data/trigger_anonymous_1_poisoned_data.pkl
# or 
python3 eval_defense.py data/anonymous_1_poisoned_data.h5 models/anonymous_1_bd_net.h5


python3 eval_defense.py data/sunglasses_poisoned_data.h5 models/sunglasses_bd_net.h5

python3 eval_defense.py "data/Multi-trigger Multi-target/eyebrows_poisoned_data.h5" models/multi_trigger_multi_target_bd_net.h5
python3 eval_defense.py "data/Multi-trigger Multi-target/lipstick_poisoned_data.h5" models/multi_trigger_multi_target_bd_net.h5
python3 eval_defense.py "data/Multi-trigger Multi-target/sunglasses_poisoned_data.h5" models/multi_trigger_multi_target_bd_net.h5

python3 eval_defense_nodata.py nodata models/anonymous_2_bd_net.h5

# test on anonymous_2_bd_net if you have data for it
# python3 eval_defensea.py the_data_for_anonymous_2_bd_net models/anonymous_2_bd_net.h5