# ML for Cyber Security
Project Report\
CSAW-HackML-2020\
Name: Eugene Wang\
Net-ID: yjw259\
https://github.com/y56/CSAW-HackML-2020/blob/master/report.md
## Introduction
In this lab we are given backdoored CNNs (called bad-net/bd_model) with known architecture (refer to `architecture.py`) and we want to "repair" the bad-net. Imagine we are buying service to train a model for us, or using some unknown source of model. Attackers may train the model to perform normally on "clean" data while output misleading result on "poisoned data" with "trigger" it.
## Methodology
Following the method in ***Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks***, I use fine-pruning as the defense approach. Although the authors also mention the pruning-aware attack. I assume the bad net we considered as just baseline attack.
### Pruning
I reset the the 77% lowest contribution neurons under clean input. The idea behind is that we believe the backdoor pattern will be capture in the `conv_3` layer but its contribution will be low when clean input are processed.
### Fine-tuning 
In this step, I retrain the model on clean input. As mentioned in the paper, this is a technique of transfer learning. We can view it as we drop some suspicious rules in the bad net first, and then train the model based on the detoxified (yet less powerful) model. Note that those reset neurons will gain weights/bias again. I save fine-pruned models for each bd model.
### Combining
I use a function `accuracy_calculator_for_combined_models()` to call both bad net and fine-pruned net and compare their outputs. An input will be considered clean if the two models give the same result, otherwise, backdoored.

### Detecting backdoored data for evaluating performance
:::warning
Actually no need for this since the provided posioned data are completely backdoored.
:::
For `anonymous_1_poisoned_data.h5` I use number of purple `(128,255,255)` pixels to detect backdoored data. This only for evaluation. I am not using this information to do the repairing. I use `check_anonymous_1_poisoned_data.py` to label backdoored inputs of `anonymous_1_poisoned_data.data` such that I can calculated  *accuracy on clean data*, *attack success rate on backdoored data*, and *attack detection rate*. 

## Discussion
### anonymous_1_bd_net
```
python3 eval_defense.py data/anonymous_1_poisoned_data.h5 models/anonymous_1_bd_net.h5 data/trigger_anonymous_1_poisoned_data.pkl
```
* before repair
    * bad net on clean validation data:
        * acc: 97.18
    * bad net on clean test data:
        * acc: 97.19
    * bad net on its corresponding poisoned data:
        * acc: 91.40
* pruned 
    * pruned bad net on clean validation data:
        * acc: 36.81
    * pruned bad net on clean test data: acc:
        * acc: 37.28
    * pruned bad net on its corresponding poisoned data:
        * acc: 50.08
* tuned and pruned 
    * tuned pruned bad net on clean validation data: 
        * acc:  99.89
    * tuned pruned bad net on clean test data: 
        * acc:  95.259
    * tuned pruned bad net on its corresponding poisoned data: 
        * acc:  8.37
* repaired (by comparing)
    * repaired net on clean validation data: 
        * acc: 97.13
        * inferred attack_ratio: 2.80 (true as 0)
    * repaired net on clean test data:
        * * acc: 93.84
        * inferred attack_ratio: 5.64 (true as 0)
    * repaired net on its corresponding poisoned data:
        * acc: 8.35
        * inferred attack_ratio: 83.66

* true attack ratio (using purple detection) : 
    * 90.93
* accuracy on those clean data within poisoned data:
    * 0.21
        * I don\'t know why so low, maybe all data od `anonymous_1_poisoned_data.h5` are backdoored? So actually no clean data in it.
        * All data in `anonymous_1_poisoned_data.h5` are labeled as zero. So, very possible there are no clean data in it.
* detected rate for those truly bd data within poisoned data:
    * 92.14
* attack succes for those clean data wihun poisoned data:
    * 7.67
 
## Reference 

Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks

Kang Liu, Brendan Dolan-Gavitt, Siddharth Garg

https://arxiv.org/abs/1805.12185

https://github.com/kangliucn


## log
### sunglasses_bd_net
```bash
python3 eval_defense.py data/sunglasses_poisoned_data.h5 models/sunglasses_bd_net.h5
```
```python=
bad net on clean validation data: acc:  97.88689702953148
bad net on clean test data: acc:  97.77864380358535
bad net on its corresponding poisoned data: acc:  99.99220576773187
pruned bad net on clean validation data: acc:  35.51571836840738
pruned bad net on clean test data: acc:  35.861262665627436
pruned bad net on its corresponding poisoned data: acc:  93.4684333593141
tuned pruned bad net on clean validation data: acc:  99.76617303195636
tuned pruned bad net on clean test data: acc:  93.4684333593141
tuned pruned bad net on its corresponding poisoned data: acc:  9.540140296180827
repaired net on clean validation data: acc, inferred attack_ratio:  (97.70503160994197, 2.260327357755261)
repaired net on clean test data: acc, inferred attack_ratio  (92.79033515198752, 6.609508963367109)
repaired net on its corresponding poisoned data: overall acc, inferred attack_ratio:  (9.540140296180827, 90.45206547155105)
```
### multi_trigger_multi_target_bd_net
#### eyebrows_poisoned_data
```bash
python3 eval_defense.py "data/Multi-trigger Multi-target/eyebrows_poisoned_data.h5" models/multi_trigger_multi_target_bd_net.h5
```
```python=
bad net on clean validation data: acc:  96.26742876937733
bad net on clean test data: acc:  96.00935307872174
bad net on its corresponding poisoned data: acc:  91.34840218238503
pruned bad net on clean validation data: acc:  46.03793193037152
pruned bad net on clean test data: acc:  45.74434918160561
pruned bad net on its corresponding poisoned data: acc:  74.29851909586905
tuned pruned bad net on clean validation data: acc:  99.91339741924308
tuned pruned bad net on clean test data: acc:  95.22213561964146
tuned pruned bad net on its corresponding poisoned data: acc:  58.47622759158223
repaired net on clean validation data: acc, inferred attack_ratio:  (96.25010825322595, 3.6806096821685284)
repaired net on clean test data: acc, inferred attack_ratio  (93.03975058456741, 6.266562743569759)
repaired net on its corresponding poisoned data: overall acc, inferred attack_ratio:  (58.44699922057678, 34.343335931410756)
``` 
#### lipstick_poisoned_data
```bash
python3 eval_defense.py "data/Multi-trigger Multi-target/lipstick_poisoned_data.h5" models/multi_trigger_multi_target_bd_net.h5 
```
```python=
bad net on clean validation data: acc:  96.26742876937733
bad net on clean test data: acc:  96.00935307872174
bad net on its corresponding poisoned data: acc:  91.52377240841777
pruned bad net on clean validation data: acc:  46.03793193037152
pruned bad net on clean test data: acc:  45.74434918160561
pruned bad net on its corresponding poisoned data: acc:  27.572096648480127
tuned pruned bad net on clean validation data: acc:  99.96535896769724
tuned pruned bad net on clean test data: acc:  95.17537022603274
tuned pruned bad net on its corresponding poisoned data: acc:  0.9840218238503509
repaired net on clean validation data: acc, inferred attack_ratio:  (96.24144799515025, 3.749891746774054)
repaired net on clean test data: acc, inferred attack_ratio  (93.02416212003118, 6.266562743569759)
repaired net on its corresponding poisoned data: overall acc, inferred attack_ratio:  (0.9840218238503509, 91.69914263445051)
```
#### sunglasses_poisoned_data
```bash
python3 eval_defense.py "data/Multi-trigger Multi-target/sunglasses_poisoned_data.h5" models/multi_trigger_multi_target_bd_net.h5
```
```python=
bad net on clean validation data: acc:  96.26742876937733
bad net on clean test data: acc:  96.00935307872174
bad net on its corresponding poisoned data: acc:  100.0
pruned bad net on clean validation data: acc:  46.03793193037152
pruned bad net on clean test data: acc:  45.74434918160561
pruned bad net on its corresponding poisoned data: acc:  0.009742790335151987
tuned pruned bad net on clean validation data: acc:  99.89607690309171
tuned pruned bad net on clean test data: acc:  95.18316445830087
tuned pruned bad net on its corresponding poisoned data: acc:  0.13639906469212784
repaired net on clean validation data: acc, inferred attack_ratio:  (96.2068069628475, 3.7585520048497445)
repaired net on clean test data: acc, inferred attack_ratio  (92.98519095869057, 6.360093530787217)
repaired net on its corresponding poisoned data: overall acc, inferred attack_ratio:  (0.13639906469212784, 99.86360093530787)
```

### anonymous_2_bd_net
```bash
python3 eval_defense_nodata.py nodata models/anonymous_2_bd_net.h5
```
```python=
bad net on clean validation data: acc:  95.82575560751711
bad net on clean test data: acc:  95.96258768511302
pruned bad net on clean validation data: acc:  36.780116047458215
pruned bad net on clean test data: acc:  37.44349181605612
tuned pruned bad net on clean validation data: acc:  99.91339741924308
tuned pruned bad net on clean test data: acc:  94.94933749025721
repaired net on clean validation data: acc, inferred attack_ratio:  (95.79111457521434, 0.04156923876331515)
repaired net on clean test data: acc, inferred attack_ratio  (92.64224473889323, 0.06851130163678877)
```
