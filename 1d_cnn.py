import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.layers import Conv1D, MaxPooling1D
# from tensorflow.keras.layers import LSTM, TimeDistributed, ConvLSTM2D
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


def load_dataset(data_rootdir, dirname, group):
    '''
    该函数实现将训练数据或测试数据文件列表堆叠为三维数组
    '''
    filename_list = []
    filepath_list = []
    X = []

    # os.walk() 方法是一个简单易用的文件、目录遍历器，可以高效的处理文件、目录。
    for rootdir, dirnames, filenames in os.walk(data_rootdir + dirname):
        for filename in filenames:
            filename_list.append(filename)
            filepath_list.append(os.path.join(rootdir, filename))
        # print(filename_list)
        # print(filepath_list)

    # 遍历根目录下的文件，并读取为DataFrame格式；
    for filepath in filepath_list:
        X.append(load_file(filepath))

    X = np.dstack(X)  # dstack沿第三个维度叠加，两个二维数组叠加后，前两个维度尺寸不变，第三个维度增加；
    y = load_file(data_rootdir + '/y_' + group + '.txt')
    # one-hot编码。这个之前的文章中提到了，因为原数据集标签从1开始，而one-hot编码从0开始，所以要先减去1
    y = to_categorical(y - 1)
    print('{}_X.shape:{},{}_y.shape:{}\n'.format(group, X.shape, group, y.shape))
    return X, y


def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 25, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    n_steps, n_length = 1, 128
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))

    model = Sequential()

    model.add(tf.keras.layers.Conv2D(126, (1, 16), input_shape=(1, 128, 9), activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    # model.add(TimeDistributed(Flatten()))
    # model.add(LSTM(100))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy



def run_experiment(trainX, trainy, testX, testy, repeats=10):
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)

    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


if __name__ == '__main__':
    train_dir = 'D:/xiangmu/UCI HAR Dataset/train/'
    test_dir =  'D:/xiangmu/UCI HAR Dataset/test/'
    dirname = '/Inertial Signals/'
    trainX, trainy = load_dataset(train_dir, dirname, 'train')
    testX, testy = load_dataset(test_dir, dirname, 'test')

    run_experiment(trainX, trainy, testX, testy, repeats=2)
