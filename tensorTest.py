
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, Flatten, Conv2D,Activation
from tensorflow.keras import datasets, layers, models
from tensorflow.python.client import device_lib
from datetime import datetime
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(
    label_mode='fine'
)

train_images, test_images = train_images / 255.0, test_images / 255.0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print(tf.config.list_physical_devices('GPU'))

def conv_layergen(model,ksize,inputsize,acti):
    model.add(layers.Conv2D(inputsize, (ksize, ksize),padding="SAME",strides=1))
    model.add(layers.Activation(acti))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2),padding="SAME"))

    return model

def conv_layergen1(model,ksize,inputsize,acti):
    model.add(layers.Conv2D(inputsize, (ksize, ksize),padding="SAME",strides=2))
    model.add(layers.Conv2D(inputsize, (ksize, ksize),padding="SAME",strides=1))
    model.add(layers.Activation(acti))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2),padding="SAME"))

    return model

def conv_layergen2(model,ksize,inputsize,acti):
    model.add(layers.Conv2D(inputsize, (ksize, ksize),padding="SAME",strides=2))

    model.add(layers.Conv2D(inputsize, (ksize, ksize),padding="SAME",strides=1))

    model.add(layers.Conv2D(inputsize, (ksize, ksize),padding="SAME",strides=1))
    model.add(layers.Activation(acti))
    model.add(layers.MaxPooling2D((2, 2),padding="SAME"))


    return model


def model_gen(conv_selection_one,conv_filter_selection_one,a_type,dropout_rate_choice,convlayers_feature):
    activation_options = ['relu','elu','relu','elu']
    conv_options = [2,3,5,7]
    output_options =[16,32,64,64,128,256]
    dropout_rates = [0.3,0.5,0.4,0.5]
    layer_levels = [1,2,3,4,5]

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(32, 32, 3)))

    valoffset = layer_levels[conv_selection_one]/10;

    for n in range(1,layer_levels[conv_selection_one]+1):
        if convlayers_feature[n] == "1":
            model = conv_layergen(model,conv_options[conv_filter_selection_one],output_options[n]*(conv_selection_one+1)//2,activation_options[a_type])
        else:
            model = conv_layergen1(model,conv_options[conv_filter_selection_one],output_options[n]*(conv_selection_one+1)//2,activation_options[a_type])

        print()
        print("--------------------------------------------------------------------------------------------------------------")
        print()
        model.add(layers.Dropout(0.25))


    model.add(layers.Flatten())

    shp = model.output_shape

    model.add(Dense(2048,activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(Dense(1024,activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(Dense(100, activation='softmax'))

    return model

def compile_model(conv_selection_one,conv_filter_selection_one,a_type,fully_connected_size,c_size2,a_type2,o_size3,c_size3,a_type3,pool1,dropout_rate_choice,convlayers_feature):

    model = model_gen(conv_selection_one,conv_filter_selection_one,a_type,dropout_rate_choice,convlayers_feature)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.summary()

    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    hist = model.fit(train_images, train_labels, epochs=2, batch_size = 32,
                        validation_data=(test_images, test_labels),
                        callbacks=[tensorboard_callback])


    model.save('saved_model/my_model')

    #plt.plot(hist.history['val_accuracy'])]

    #plt.title('Validation accuracy data')
    #plt.ylabel('validation accuracy value')
    #plt.xlabel('No. epoch')
    #plt.show()
    val = hist.history['val_accuracy']
    tf.keras.backend.clear_session()

    return val




#compile_model(2,0,1,1,0,0,0,0,0,0,0,0)
