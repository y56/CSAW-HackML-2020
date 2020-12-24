import h5py # The h5py package is a Pythonic interface to the HDF5 binary data format.
import keras
import keras.backend as K
import numpy as np
import sys
import tensorflow as tf


def data_loader(filepath): # from eval.py
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))
    return x_data, y_data

def pruned(model, layer_name, clean_validation_data_x):


    # print("==== build a partial model, containing from input_layer to conv_3_layer")
    # get conv_3's output
    # ref: https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
    # ref: https://keras.io/api/models/model/
    modelinput = model.input
    modelouputs= [layer.output for layer in model.layers]
    conv3output = modelouputs[5]  # the conv_3 layer # we can get layer by name by `model.get_layer(layer_name)`
    partial_model = keras.models.Model(inputs=modelinput, outputs=conv3output) # give us a truncated network
    # the connections of all in-between layers are already defined
    # specifying the input/output layer is sufficient to give a model
    # ref: https://keras.io/api/models/model/
    # print("partial model info:")
    # print(partial_model.summary())
    # print("conv_3_layer info:")
    # print(conv3output)


    # print("=== compare between neurons' total output")
    conv3output_outputvalue = partial_model.predict(clean_validation_data_x)
    ##print(conv3output_outputvalue)
    # print("number of neurons to compare:", conv3output_outputvalue.shape[3]) # the last dim
    numof_conv3output = conv3output_outputvalue.shape[3] # np.shape(conv3output_outputvalue)[-1]
    sample_num = np.shape(conv3output_outputvalue)[0]
    ##print(np.shape(conv3output_outputvalue))

    conv3output_sum = [np.sum(conv3output_outputvalue[:, :, :, _i]) for _i in range(numof_conv3output)]
    ##print("each conv3output's sum")
    ##print(conv3output_sum)
    idx_sorted_by_sum = sorted([(_val, _idx) for _idx,_val in enumerate(conv3output_sum) ])
    # sorted by val, yet we only need index
    idx_sorted_by_sum = [_tuple[1] for _tuple in idx_sorted_by_sum]
    # print("indices, output's sum from min to max:",idx_sorted_by_sum)


    # record original neurons info
    orig_weights, orig_bias = model.get_layer(layer_name).get_weights()


    
    numof_neurons_to_prune = int(len(idx_sorted_by_sum) * PRUNING_RATIO) # paper fig.6(a), good when 75% neurons are pruned
    # paper: https://arxiv.org/abs/1805.12185
    for _i in range(numof_neurons_to_prune):
        idx_for_neuron=idx_sorted_by_sum[_i]
        orig_weights[:, :, :, idx_for_neuron] = np.zeros(np.shape(orig_weights[:, :, :, idx_for_neuron]))
        orig_bias[idx_for_neuron] = 0
    model.get_layer(name=layer_name).set_weights((orig_weights, orig_bias))


    # print('=== delete number of nuerons: ', numof_neurons_to_prune)

    return model

def tuned(model, x, y):
    model.compile(keras.optimizers.Adam(lr=1e-3, amsgrad=True), # Configures the model for training.
        # AMSGrad
        # ref: https://paperswithcode.com/method/amsgrad
        loss="categorical_crossentropy", metrics=["accuracy"])
    y_onehot = keras.utils.to_categorical(y, num_classes=1283) # encode label into onehot
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
    model.fit(x, y_onehot, epochs=5, batch_size=200) # clean validation has 11547 samples
    # The batch size is a number of samples processed before the model is updated.
    return model
    
def accuracy_calculator(x_test, y_test, model):
    clean_label_p = np.argmax(model.predict(x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test)) * 100
    return class_accu

def accuracy_calculator_for_combined_models(x, y, bd_model, tuned_pruned_model):
    bd_result = np.argmax(bd_model.predict(x), axis=1)
    # print(bd_result)
    # print(bd_result.shape)
    tp_result = np.argmax(tuned_pruned_model.predict(x), axis=1) # tp for tuned_pruned
    # print(tp_result)
    # print(tp_result.shape)
    attack_ct=0
    repaired_result = np.zeros(np.shape(y))
    for i, (bd, tp) in enumerate(zip(bd_result, tp_result)):
        # print(i, (bd, tp)s)
        if bd==tp:
            repaired_result[i]=tp
        else:
            attack_ct+=1
            repaired_result[i]=1283 # 0~1282 are normal, 1283 are attacks
    # print(repaired_result)     
    class_accu = np.mean(np.equal(repaired_result, y)) * 100
    return class_accu, attack_ct/len(y)


if __name__ == '__main__':

    # constant
    PRUNING_RATIO = 0.77 # naive constaint # just drop a certain ratio of neurons
    # we can do it in a more sophisticated way:
    # use acc threshold and calculate acc after we drop one more neuron

    # (poisoned) data path and bad net model path from command line arguement
    # input_data_dir = str(sys.argv[1])
    bd_model = keras.models.load_model(str(sys.argv[2]))
    # input_data_x, input_data_y = data_loader(input_data_dir)
    # input_data_x /= 255

    # clena data I will use for pruning
    clean_validation_data_dir = 'data/clean_validation_data.h5' # clean data to check conv_3 to prune
    clean_validation_data_x, clean_validation_data_y = data_loader(clean_validation_data_dir)
    clean_validation_data_x /= 255 # pixel intensity into 0~1

    # clean data I will use to check accuracy for bad net and repaired bad net
    clean_test_data_dir = 'data/clean_test_data.h5' # clean data to check conv_3 to prune
    clean_test_data_x, clean_test_data_y = data_loader(clean_test_data_dir)
    clean_test_data_x /= 255 # pixel intensity into 0~1

    print("bad net on clean validation data: acc: ", accuracy_calculator(clean_validation_data_x, clean_validation_data_y, bd_model))
    # print("y",len(set(clean_validation_data_y)), min(clean_validation_data_y), max(clean_validation_data_y))
    print("bad net on clean test data: acc: ", accuracy_calculator(clean_test_data_x, clean_test_data_y, bd_model))
    # print("y",len(set(clean_test_data_y)), min(clean_test_data_y), max(clean_test_data_y))
    # print("bad net on its corresponding poisoned data: acc: ", accuracy_calculator(input_data_x, input_data_y, bd_model))
    # print("y",len(set(input_data_y)), min(input_data_y), max(input_data_y))

    # prune the bad net
    pruned_model = pruned(bd_model, "conv_3", clean_validation_data_x)

    print("pruned bad net on clean validation data: acc: ", accuracy_calculator(clean_validation_data_x, clean_validation_data_y, pruned_model))
    print("pruned bad net on clean test data: acc: ", accuracy_calculator(clean_test_data_x, clean_test_data_y, pruned_model))
    # print("pruned bad net on its corresponding poisoned data: acc: ", accuracy_calculator(input_data_x, input_data_y, pruned_model))
    
    # tune the pruned bad net
    tuned_pruned_model   = tuned(pruned_model, clean_validation_data_x, clean_validation_data_y)

    print("tuned pruned bad net on clean validation data: acc: ", accuracy_calculator(clean_validation_data_x, clean_validation_data_y, tuned_pruned_model))
    print("tuned pruned bad net on clean test data: acc: ", accuracy_calculator(clean_test_data_x, clean_test_data_y, tuned_pruned_model))
    # print("tuned pruned bad net on its corresponding poisoned data: acc: ", accuracy_calculator(input_data_x, input_data_y, tuned_pruned_model))


    # python passes by ref
    # so... tuned_pruned_model and bd_model are now referring to the same obj...
    # load bd_model again here
    bd_model = keras.models.load_model(str(sys.argv[2]))

    # combine bad-net and tuned-pruned-net
    print("repaired net on clean validation data: acc, inferred attack_ratio: ",
        accuracy_calculator_for_combined_models(
        clean_validation_data_x, clean_validation_data_y, bd_model, tuned_pruned_model))
    print("repaired net on clean test data: acc, inferred attack_ratio ",
        accuracy_calculator_for_combined_models(
        clean_test_data_x, clean_test_data_y, bd_model, tuned_pruned_model))
    # print("repaired net on its corresponding poisoned data: attack success rate, inferred attack_ratio: ", 
        # accuracy_calculator_for_combined_models(
        # input_data_x, input_data_y, bd_model, tuned_pruned_model))