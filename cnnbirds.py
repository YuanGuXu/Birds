import matplotlib
import tensorflow as tf
import tensorflow as keras
import numpy as np
import glob
import random
import plistlib
from matplotlib import pyplot as plt
from tensorflow.keras.layers import LSTM
import keras.backend as K


imags_path=glob.glob('birds/*/*.jpg')#路径

#获取名称
# print(imags_path[:3])
# img_p=imags_path[100]
# print(img_p)
# print(img_p.split('\\')[1].split('.')[1])


all_labels_name=[img_p.split('\\')[1].split('.')[1] for img_p in imags_path]#获取全部名字:例如Black_footed_Albatross
label_name=np.unique(all_labels_name)#名字唯一值：共200个
label_to_index=dict((name,i) for i,name in enumerate(label_name))#字典：名字————序号
# print(label_to_index)
index_to_lable=dict((v,k) for k,v in label_to_index.items())#字典：序号————名字
# print(index_to_lable)
all_labels=[label_to_index.get(name) for name in all_labels_name]
#print(all_labels[-3:]) #[54, 54, 54]



np.random.seed(2021)#乱序
random.index=np.random.permutation(len(imags_path))
imgs_path=np.array(---------------------imags_path)[random.index]
all_labels=np.array(all_labels)[random.index]



#划分出训练集、测试集
i =int(len(imags_path)*0.8)
train_path=imags_path[:i]
train_labels=all_labels[:i]
test_path=imgs_path[i:]
test_labels=all_labels[i:]
#print(train_path) #'birds\\001.Black_footed_Albatross\\Black_Footed_Albatross_0001_796111.jpg'

#创建datest
train_ds=tf.data.Dataset.from_tensor_slices((train_path,train_labels))#注意：这里是路径
test_ds=tf.data.Dataset.from_tensor_slices((test_path,test_labels))

#加载图片的函数
def load_img(path,label):
    image=tf.io.read_file(path)
    image=tf.image.decode_jpeg(image,channels=3)
    image=tf.image.resize(image,[256,256])
    image=tf.cast(image,tf.float32)
    image=image/255
    return image,label


AUTOTUNE=tf.data.experimental.AUTOTUNE#自动选择线程数
train_ds=train_ds.map(load_img,num_parallel_calls=AUTOTUNE)
test_ds=test_ds.map(load_img,num_parallel_calls=AUTOTUNE)
BATCH_SIZE=8
train_ds=train_ds.repeat().shuffle(200).batch(BATCH_SIZE)
test_ds=test_ds.batch(BATCH_SIZE)

#建立模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'))#64卷积核数目；（3，3），过滤器大小
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(200, activation='softmax'))


# model.summary()#将信息打印到屏幕终端

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['acc']
              )

train_count = len(train_path)
test_count = len(test_path)
step_per_epoch = train_count//BATCH_SIZE
validation_step = test_count//BATCH_SIZE


history = model.fit(train_ds, epochs=5, steps_per_epoch=step_per_epoch,
                        validation_data=test_ds,
                        validation_steps=validation_step
                        )

plt.figure()
plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
plt.show()
plt.figure()
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.show()



#使用模型进行预测
def load_and_preprocess_image(path):
    image=tf.io.read_file(path)
    image=tf.image.decode_jpeg(image,channels=3)
    image=tf.image.resize(image,[256.256])
    image=tf.cast(image,tf.float32)
    image=image/255.0
    return image
test_img='Black_Footed_Albatross_0001_796111.jpg'
test_tensor=load_and_preprocess_image(test_img)
test_tensor=tf.expand_dims(test_tensor,axis=0)#扩展维度，因为模型需要在一个epoch上运行
pred=model.predict(test_tensor)#pred为长度为200的张量
index_to_lable.get(np.argmax(pred))#返回张量中最大值的位置


