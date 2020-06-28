import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from texttable import Texttable
from sklearn.model_selection import KFold
from typing import List


class DataPreperation:

    def convert_image(img):
        image = tf.image.convert_image_dtype(img, tf.float32) # Cast and normalize the image to [0,1]
        return image

    def augment(img, label,height, width, depth):
        image_copy = img.copy()
        image = DataPreperation.convert_image(image_copy)  # Cast and normalize the image to [0,1]
        tf.keras.preprocessing.image.random_rotation(image, 180, row_axis=1, col_axis=2, channel_axis=0,
                                                     fill_mode='nearest', cval=0.0,
                                                     interpolation_order=1)
        #image = tf.image.resize_with_crop_or_pad(image, hight+6, width+6)  # Add 6 pixels of padding

        crop_scale = np.random.uniform(low=1.0, high=2.5 , size=1)
        image = tf.image.random_crop(image, size=[int(height//crop_scale[0]), int(width//crop_scale[0]), depth])  # image crop loose the dimension
        image = tf.image.resize(image, [width,height]) # resize the image
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_contrast(image, lower=0.5, upper=0.8)

        #image = tf.image.random_brightness(image, max_delta=0.1)  # Random brightness :it cause pixels to be out of range 0,255
        # To have the same format for all the images :
        # If you want to go on GPU dont convert the data
        #image = tf.keras.preprocessing.image.img_to_array(image)

        return image, label

    def Kfold_data(X,Y,split_num = 10):

        for train_index, val_index in KFold(n_splits=split_num, shuffle=False).split(X):
            x_train, x_val = X[train_index], X[val_index]
            y_train, y_val = Y[train_index], Y[val_index]

            yield x_train, x_val,y_train, y_val


class DataVisualization:

    def files_reading_summary(Folder_names: List[str], Total_images_num: int, Total_label_num: int,
                              To_print_list: List[List[str]]) -> None:
        """

        """
        Total_class_num = len(Folder_names)
        print("\n________Table : Summary _________")
        t = Texttable()
        t.add_rows([
            ['# Images', ' # Labels', '# Classes'],
            [Total_images_num, Total_label_num, Total_class_num]
        ])
        print(t.draw())
        print("//////////////////////////////////////////////////\n")

        print("_______________________Table : Data Folders Reading info________________________")
        t = Texttable()
        for i in range(1, len(To_print_list)):
            To_print_list[i].insert(2, "{:.2%}".format(To_print_list[i][1] / Total_images_num))

        t.add_rows(To_print_list)
        print(t.draw())
        print("//////////////////////////////////////////////////\n")


    def print_one_set_statistics(x_train: np.array, y_train: np.array) -> None:
        """

        """
        Total_data_num = len(x_train)
        print("\n________Table : Data portions info_________")
        t = Texttable()
        t.add_rows([
            ['Data Portion', 'Number', 'Percent'],
            ['Total', Total_data_num, "{:.0%}".format(1)],
        ])
        print(t.draw())
        print("//////////////////////////////////////////////////\n")

        print("_______________________Table : Data shape info________________________")
        t = Texttable()
        t.add_rows([
            ['Name', 'Shape', 'Min', 'Max', 'Type'],
            ['x train', x_train.shape, x_train.min(), x_train.max(), type(x_train)],
            ['y train', y_train.shape, y_train.min(), y_train.max(), type(y_train)]
        ])
        print(t.draw())
        print("//////////////////////////////////////////////////\n")

    def print_data_statistics(x_train: np.array, x_val: np.array, x_test: np.array, y_train: np.array, y_val: np.array) -> None:
        """

        """
        Total_data_num = len(x_train) + len(x_val) + len(x_test)
        print("\n________Table : Data portions info_________")
        t = Texttable()
        t.add_rows([
            ['Data Portion', 'Number', 'Percent'],
            ['Total', Total_data_num, "{:.0%}".format(1)],
            ['Train data', len(x_train), "{:.2%}".format(len(x_train) / Total_data_num)],
            ['val data', len(x_val), "{:.2%}".format(len(x_val) / Total_data_num)],
            ['Test data', len(x_test), "{:.2%}".format(len(x_test) / Total_data_num)]
        ])
        print(t.draw())
        print("//////////////////////////////////////////////////\n")

        print("_______________________Table : Data shape info________________________")
        t = Texttable()
        t.add_rows([
            ['Name', 'Shape', 'Min', 'Max', 'Type'],
            ['x train', x_train.shape, x_train.min(), x_train.max(), type(x_train)],
            ['y train', y_train.shape, y_train.min(), y_train.max(), type(y_train)],
            ['x validation', x_val.shape, x_val.min(), x_val.max(), type(x_val)],
            ['y validation', y_val.shape, y_val.min(), y_val.max(), type(y_val)],
            ['x test', x_test.shape, x_test.min(), x_test.max(), type(x_test)]
        ])
        print(t.draw())
        print("//////////////////////////////////////////////////\n")

    def show_image_plt(title: str, train_images, train_labels, class_names, raw_num=8, max=255, probabilites=None):
        fig = plt.figure(figsize=(10, 10))
        plt.subplots_adjust(left=0.02, bottom=0.03, right=0.98, top=0.99, wspace=None, hspace=None)
        fig.canvas.set_window_title(title)

        for i in range(raw_num * raw_num):
            plt.subplot(raw_num, raw_num, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            if train_images[i].shape[-1] == 3:
                plt.imshow(train_images[i])#cmap=plt.cm.binary
            else:
                plt.imshow(train_images[i], cmap='gray', vmin=0, vmax=max)

            if probabilites is None:
                plt.xlabel(class_names[train_labels[i]])

            '''
            else :
                ss = train_labels[i][0]
                string_1 = class_names[train_labels[i][0]]
                string_2 = int(probabilites[i][0]*100)
                plt.xlabel(class_names[train_labels[i][0]] + '(' + str(int(probabilites[i][0]*100))+')')
            '''

        plt.show()

    def show_learning_status(process):
        # ///////////////////// Visualize the training process ////////////////////
        train_loss_history = process.history['loss']
        train_acc_history = process.history['accuracy']
        val_loss_history = process.history['val_loss']
        val_acc_history = process.history['val_accuracy']
        # print('\nTrain loss: ',train_loss_history)
        # print('\nTrain acc: ',train_acc_history)
        fig = plt.figure(figsize=(10, 10))
        fig.canvas.set_window_title('Learning Status')
        ax1, ax2 = plt.subplot(2, 1, 1), plt.subplot(2, 1, 2)

        x1 = np.arange(0, len(train_loss_history), 1)
        y1 = np.array(train_loss_history, dtype=np.float32)
        y2 = np.array(train_acc_history, dtype=np.float32)
        y3 = np.array(val_loss_history, dtype=np.float32)
        y4 = np.array(val_acc_history, dtype=np.float32)

        ax1.plot(x1, y1, 'o-', label="Training")
        ax1.plot(x1, y3, 'o-', label="Validation")
        ax2.plot(x1, y2, 'o-', label="Training")
        ax2.plot(x1, y4, 'o-', label="Validation")

        # naming the x axis
        ax1.set_xlabel('epoch')
        ax2.set_xlabel('epoch')
        # naming the y axis
        ax1.set_ylabel('loss')
        ax2.set_ylabel('Accuracy')
        # giving a title to my graph
        ax1.set_title('Loss function')
        ax2.set_title('Accuracy')
        ax1.grid()  # show grid
        ax2.grid()  # show grid
        ax1.legend()  # show legend
        ax2.legend()  # show legend
        plt.show()