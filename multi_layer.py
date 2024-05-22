import numpy as np
import keras
# from keras import layers
import matplotlib.pyplot as plt

def createGraph(history,num_of_layer):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Gaph of the model accuracy with ' + str(num_of_layer) + ' layers')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train data', 'validation data'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Gaph of the model loss with ' + str(num_of_layer) + ' layers')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train data', 'validation data'], loc='upper right')
    plt.show()


def main():
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 1

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == '__main__':
    main()







