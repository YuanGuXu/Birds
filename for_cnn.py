import numpy as np
def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in
             [row.replace('  ', ' ').strip().split(' ') for row in file]])
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))
def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.int32)
    file.close()
    return y_ - 1

def one_hot(y_):
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]




if __name__ == '__main__':
    """
    if __name__ == '__main__': 的作用
    一个python文件通常有两种使用方法，第一是作为脚本直接执行，第二是
    import 到其他的python脚本中被调用（模块重用）执行。因此if __name__ == 'main': 的作用就是控制这两种情况执行代码的过程，
    在if __name__ == 'main': 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行，而
    import 到其他脚本中是不会被执行的
    """

    # 指定九种不同输入信号，即9个文件的文件名前缀
    INPUT_SIGNAL_TYPES = [
        'body_acc_x_',
        'body_acc_y_',
        'body_acc_z_',
        'body_gyro_x_',
        'body_gyro_y_',
        'body_gyro_z_',
        'total_acc_x_',
        'total_acc_y_',
        'total_acc_z_'
    ]

    # 六种行为标签，行走 站立 躺下 坐下 上楼 下楼
    LABELS = [
        'WALKING',
        'WALKING_UPSTAIRS',
        'WALKING_DOWNSTAIRS',
        'SITTING',
        'STANDING',
        'LAYING'
    ]

    # 指定数据路径
    DATASET_PATH ='D:/xiangmu/UCI HAR Dataset/'
    print('\n' + 'Dataset is now located at:' + DATASET_PATH)
    TRAIN = 'train/'
    TEST = 'test/'

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + 'Inertial Signals/' + signal + 'train.txt' for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [DATASET_PATH + TEST + 'Inertial Signals/' + signal + 'test.txt' for signal in
                            INPUT_SIGNAL_TYPES]



#*************************************************************************

    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)
    print(X_train.shape)
    X_train = X_train.reshape((7352,4,32,9))
    print(X_train[0])


# p=np.array([
#     [11,12,13,14,15,16],
#     [21,22,23,24,25,26],
#     [31,32,33,34,35,36],
#     [41,42,43,44,45,46]
# ])
#
# # print(p.shape)
#
# p=np.transpose(np.array(p), (1,0))
# print(p)
# p=p.reshape(2,3,4)
#
# print(p.shape)
# print(p)