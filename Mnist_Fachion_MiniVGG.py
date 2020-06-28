import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from mnist import download_and_parse_mnist_file
from helper_functions import DataPreperation, DataVisualization


# Prevent Memmory overflow in GPU
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_dataset() -> (np.array, np.array, np.array, np.array, np.ndarray):
    """

    """
    path = 'dataset/Mnist_Fashion'
    x_train_file, y_train_file = 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte'
    x_test, y_test = 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'
    data_names = [x_train_file, y_train_file, x_test, y_test]
    Data = [download_and_parse_mnist_file(fname, target_dir=path) for fname in data_names]
    x_train, y_train, x_test, y_test = Data[0], Data[1], Data[2], Data[3]
    # Add a new dimension to images :
    # it hsould be 28,28,1 not 28 ,28
    x_train = np.expand_dims(x_train,axis = -1)
    y_train = np.expand_dims(y_train,axis = -1)
    x_test = np.expand_dims(x_test,axis = -1)
    y_test = np.expand_dims(y_test,axis = -1)

    avilable_classes = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle','boot']
    return x_train, y_train, x_test, y_test, avilable_classes




def main():
    # ////////////////////// Prepare input data ///////////////////////////
    x_train, y_train, x_test, y_test, class_names = load_dataset()
    # Reserve validation data from training data \n\n
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20)
    DataVisualization.show_image_plt("Training data set example", np.squeeze(x_train, axis=-1), y_train.flatten(),
                                     class_names)
    # Normalize input features
    x_train, x_val, x_test = x_train / 255.0,x_val/255, x_test / 255.0
    DataVisualization.print_data_statistics(x_train, x_val, x_test, y_train, y_val, y_test)



    # ///////////////////// Define the architecture ////////////////////
    model = tf.keras.models.Sequential()
    # First CONV => RELU => CONV => RELU => POOL layer set
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu',input_shape=(28,28,1)))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    # Second CONV => RELU => CONV => RELU => POOL layer set
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    # first (and only) set of FC => RELU layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    # softmax classifier
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())


    # first option  
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    '''
    # second option
    NUM_EPOCHS,INIT_LR  = 25, 1e-2
    from tensorflow.keras.optimizers import SGD
    opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
                  metrics=["accuracy"])

    print(model.summary())
    '''


    # ///////////////////// Train the Model ////////////////////
    # BS = 32
    # Epoch =25
    history = model.fit(x_train, y_train,batch_size=100, epochs= 10,validation_data=(x_val, y_val))
    DataVisualization.show_learning_status(history)

    # ///////////////////// Touch Test set :Evaluate the model  ////////////////////
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('\nTest accuracy: {} , Test loss : {}'.format(test_acc, test_loss))

    # /// Visualize first 25 test data with its classes
    Y_test_classes = np.argmax(model.predict(x_test[0:100]), axis=-1)
    DataVisualization.show_image_plt("Predicted labels on test data set", np.squeeze(x_test, axis=-1), Y_test_classes, class_names,max=1)

if __name__ =='__main__':
    main()