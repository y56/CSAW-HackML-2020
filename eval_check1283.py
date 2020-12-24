"""
to check some properties
"""

# import keras
import sys
import h5py
import numpy as np

clean_data_filename = str(sys.argv[1])
model_filename = str(sys.argv[2])


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))
    print(y_data.shape)
    for y in y_data:
        if y == 1283.0 or y == 1283:
            print(filepath)

    return x_data, y_data

# def data_preprocess(x_data):
#     return x_data/255


def main():
    x_test, y_test = data_loader(clean_data_filename)
    # x_test = data_preprocess(x_test)

    # bd_model = keras.models.load_model(model_filename)
    # clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
    # class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    # print("by model:",len(clean_label_p))
    # print("by data:",len(y_test))
    # print("by model:",set(clean_label_p))
    # print("by data:",set(y_test))
    # print("by model:",len(set(clean_label_p)))
    # print("by data:",len(set(y_test)))
    # print('Classification accuracy:', class_accu)


if __name__ == '__main__':
    main()
