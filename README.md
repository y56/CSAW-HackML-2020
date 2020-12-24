# CSAW-HackML-2020

```
├── data 
    └── clean_validation_data.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test_data.h5 // don't use it to design/train anything
    └── sunglasses_poisoned_data.h5
    └── anonymous_1_poisoned_data.h5
    └── trigger_anonymous_1_poisoned_data.pkl // a list produced by `check_anonymous_1_poisoned_data.py`// True for being backdoored
    └── Multi-trigger Multi-target
        └── eyebrows_poisoned_data.h5
        └── lipstick_poisoned_data.h5
        └── sunglasses_poisoned_data.h5
├── models / fine_pruned_models
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_bd_weights.h5
    └── anonymous_1_bd_net.h5
    └── anonymous_1_bd_weights.h5
    └── anonymous_2_bd_net.h5
    └── anonymous_2_bd_weights.h5
├── architecture.py   		// contains the names of each layer
├── eval.py           		// this is the evaluation script
├── eval_check.py     		// to chekc dimensions to do some tests
├── eval_check1283.py 		// check if there are 1283 label // thers is no such
├── eval_defense.py   		// to defense and print some statistics
├── eval_defense_nodata.py 	// when no data provided
├── show_photo.py           // show photos
├── fine_prune.py    		// fine-prune and save fine-pruned models // save at fine_pruned_models
├── fine_prune_all.sh 		// do fine-prune.py on all bd model
├── check_anonymous_1_poisoned_data.py
└── runall.sh // run for each bad model with suitable .py and data (and trigger label if any)
```

## I. Dependencies
   1. Python 3.6.9
   2. Keras 2.3.1
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. TensorFlow-gpu 1.15.2
   
## II. Validation Data
   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing) and store them under `data/` directory.
   2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals each containing 9 images in the validation dataset.
   3. sunglasses_poisoned_data.h5 contains test images with sunglasses trigger that activates the backdoor for sunglasses_bd_net.h5. Similarly, there are other .h5 files with poisoned data that correspond to different BadNets under models directory.

## III. Evaluating the Backdoored Model
   1. The DNN architecture used to train the face recognition model is the state-of-the-art DeepID network. This DNN is backdoored with multiple triggers. Each trigger is associated with its own target label. 
   2. To evaluate the backdoored model, execute `eval.py` by running:  
      `python3 eval.py <clean validation data directory> <model directory>`.
      
      E.g., `python3 eval.py data/clean_validation_data.h5  models/sunglasses_bd_net.h5`. Clean data classification accuracy on the provided validation dataset for sunglasses_bd_net.h5 is 97.87 %.

## IV. Evaluating the Submissions The teams should submit a single eval.py

script for each of the four BadNets provided to you. In other words, your
submission should include four eval.py scripts, each corresponding to one of
the four BadNets provided. YouTube face dataset has classes in range [0,
1282]. So, your eval.py script should output a class in range [0, 1283] for a
test image w.r.t. a specific backdoored model. Here, output label 1283
corresponds to poisoned test image and output label in [0, 1282] corresponds
to the model's prediction if the test image is not flagged as poisoned.
Effectively, design your eval.py with input: a test image (in png or jpeg
format), output: a class in range [0, 1283]. Output 1283 if the test image is
poisoned, else, output the class in range [0,1282].

Teams should submit their solutions using GitHub. All your models (and datasets) should be uploaded to the GitHub repository. If your method relies on any dataset with large size, then upload the data to a shareable drive and provide the link to the drive in the GitHub repository. To efficiently evaluate your work, provide a README file with clear instructions on how to run the eval.py script with an example.
For example: `python3 eval_anonymous_2.py data/test_image.png`. Here, eval_anonymous_2.py is designed for anonynous_2_bd_net.h5 model. Output should be either 1283 (if test_image.png is poisoned) or one class in range [0, 1282] (if test_image.png is not poisoned).

# for submission

## to evaluate on `anonymous_2_bd_net`

Use `anonymous_2_poisoned_data` as the same format of `anonymous_1_poisoned_data`.

If we put `path_data` for fully poisoned data, in

```
repaired net on its corresponding poisoned data: overall acc, inferred attack_ratio:
````

`overall acc` is the attack success rate.

```bash
python3 eval_defensea.py path_data models/anonymous_2_bd_net.h5
```

## to evaluate those with provided data

```bash
python3 eval_defense.py data/anonymous_1_poisoned_data.h5 models/anonymous_1_bd_net.h5
python3 eval_defense.py data/sunglasses_poisoned_data.h5 models/sunglasses_bd_net.h5
python3 eval_defense.py "data/Multi-trigger Multi-target/eyebrows_poisoned_data.h5" models/multi_trigger_multi_target_bd_net.h5
python3 eval_defense.py "data/Multi-trigger Multi-target/lipstick_poisoned_data.h5" models/multi_trigger_multi_target_bd_net.h5
python3 eval_defense.py "data/Multi-trigger Multi-target/sunglasses_poisoned_data.h5" models/multi_trigger_multi_target_bd_net.h5
```

# Project Report

## Reference 

Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks

Kang Liu, Brendan Dolan-Gavitt, Siddharth Garg

https://arxiv.org/abs/1805.12185

https://github.com/kangliucn

## Methodology

### in general
Following the paper, I assume backdoor features are captured by conv_3 neurons who make little contribution when clean inputs comes in. (referring to `architecture.py` for the stucture of bad-nets). I also assume that these attacks are not "pruning-aware" .

### pruning
* Sort conv_3 neurons by their individual total output.
* Reset about 77% smallest (weight, bias set to zeros)
    * We also can use accuracy as threshold. That needs more computing time.

### tuning

* Use clean data to train again. Note that those reset neurons will get values again.

### compare
* Compare bad-net with tuned-pruned-net. If they predict differently, mark the input as an attack. We can also use Lambda layer of tf.keras.layers.Layer to make it a portable model. (ref: https://keras.io/api/layers/core_layers/lambda/)

## misc 
To verify attack success rate etc, I should use a good net to know the correct labels and to know which data are backdoored.

Have to plot some images to have more sense.

Have to check if some data are labeled as 1283(i.e., attack). 
**Checked, no such stuff**
```
for f in data/*.h5; do python3 eval_check1283.py "$f" noneed; done;
for f in data/*/*.h5; do python3 eval_check1283.py "$f" noneed; done;
```


