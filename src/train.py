import numpy as np
import matplotlib.pyplot as plt

from keras.models       import load_model
from keras.callbacks    import ModelCheckpoint
from keras              import optimizers
from keras.callbacks    import ModelCheckpoint

from src.model          import Model
from src.utilities      import absolute_file_path
class Train(object):
    """
    A class to Train new model
    
    Attributes
    ----------

    Methods
    -------
    """
    def train(self):
        """
        Arguments
        ---------
        @arg x_train: input dataset
        @arg y_train: output dataset
        """
        print('Loading datasets')
        # load training dataset
        dataset_train = np.loadtxt(absolute_file_path('datasets/train_2dof.csv'), delimiter=",")
        x_train=dataset_train[:,:2] # (input vector) first two columns are end effector states
        y_train=dataset_train[:,2:] # (output vector) second and third columns are joint angles
        print(x_train.shape)
        # load test dataset
        dataset_test = np.loadtxt(absolute_file_path('datasets/test_2dof.csv'), delimiter=",")
        x_test=dataset_test[:,:2]
        y_test=dataset_test[:,2:]
        # prediction dataset
        dataset_predict = np.loadtxt(absolute_file_path('datasets/pred_2dof.csv'), delimiter=",")
        x_predict=dataset_predict[:,:2]

        input_dim = x_train.shape[1]
        output_dim = y_train.shape[1]
        mdl = Model().create_model(input_dim, [8,16,32,16,8], output_dim)
        mdl.summary()
        
        # Optimizer
        o = optimizers.Adagrad(lr=0.0001, epsilon=None, decay=0.0)
        ''' Possible optimisers
        o = optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
        o = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        o = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        o = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        o = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        o = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        o = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        '''
        loss_func='mse'
        mdl.compile(loss=loss_func, optimizer= o, metrics=['accuracy'])

        # saves the model weights after each epoch if the validation loss decreased
        checkpoint = ModelCheckpoint(   filepath       = absolute_file_path('models/tmp/temp_weights.hdf5'), 
                                        verbose        = True, 
                                        save_best_only = True)
        
        history = mdl.fit(  x_train,
                            y_train,
                            batch_size=64,  # Number of samples per gradient update
                            epochs=100,     # Number of epochs to train the model
                            verbose=1,      #  0 = silent, 1 = progress bar, 2 = one line per epoch.
                            validation_data=(x_test, y_test),
                            callbacks=[checkpoint])
        # Test model
        score = mdl.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # Predict output
        solution = mdl.predict(x_predict)
        # solution=model.predict(x_predict)
        print(solution)

        save_choice = input("enter 1 to save model")

        if save_choice == 1:
            modelName=input("Enter name for model")
            mdl.save(absolute_file_path('models/mdl_'+modelName+'.h5'))  # creates a HDF5 model file 

        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        del model  # deletes the existing model