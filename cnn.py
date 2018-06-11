from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten,Dropout
from keras.layers.core import Dense # fully-connected net
from keras import backend as K

# simple cnn model on cifar10 small images dataset

class Cnn:
  @staticmethod
  def build(width, height, depth, classes):
    # parameter: classes means the total number of classes we want to recognize
    #initialize the model
    model = Sequential()
    inputShape = (height, width, depth)

    # if we are using "channel first", update the input shape
    # in some situation like TH, use channel first
    if K.image_data_format() == "channel_first":
      inputShape = (depth, height, width)

    # first set of CONV => RELU => POOL layers => Dropout
    # conv layer will learn 32 convolution filters, each of which are 3*3
    model.add(Conv2D(32, (3, 3),padding = "same",input_shape = inputShape))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(64, (3, 3),padding = "same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # flattening out the volume into a set of fully-connected layer
    # first and only set of FC => RELU laters
    model.add(Flatten())
    # fully-connected layer has 512 units
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # softmax classifier (output layer)
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    #return the constructed network architecture
    return model


