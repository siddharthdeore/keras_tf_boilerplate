import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import plot_model

from src.utilities import absolute_file_path


class Test:
    """
    A class to Test existing model
    
    Attributes
    ----------

    Methods
    -------
    """
    def test(self,x_predict,y_predict):
        model = load_model(absolute_file_path('../models/model.h5'))
        model.summary()

        solution=model.predict(x_predict,verbose=2)
        print(solution)
        print(y_predict)

        #plt.plot(solution[:,0],y_predict[:,0],'ro')
        #plt.plot(y_predict[:,0],y_predict[:,1],'bo')
        #plt.show()

        del model  # deletes the existing model