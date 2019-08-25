from keras.models import Sequential
from keras.layers import Dense

class Model(object):

    def create_model(input_layer_dim=2,hidden_units=[32,64,64,32],output_layer_dim=2):
        """
        Arguments
        ---------
        @arg input_layer_dim: dimention of input layer
        @arg hidden_units: array of length eqauls number of hidden layers with each element denotes number of units in that layer. 
        @arg output_layer_dim:dimention of output layer
        """
        model = Sequential()

        # input layer
        layer = Dense(  input_dim          = input_layer_dim,
                        units              = hidden_units[0],
                        kernel_initializer = 'random_normal',
                        use_bias           = True,
                        bias_initializer   = 'random_normal',
                        activation         = 'tanh')

        model.add(layer)

        # hidden layers
        for i in range(len(hidden_units)-1):
            h_layer = Dense(  units              = hidden_units[i+1],
                                kernel_initializer = 'random_normal',
                                use_bias           = True,
                                bias_initializer   = 'random_normal',
                                activation         = 'tanh')
            model.add(h_layer)
        
        # output layer
        layer = Dense(  units              = output_layer_dim,
                        kernel_initializer = 'random_normal',
                        use_bias           = True,
                        bias_initializer   = 'random_normal',
                        activation         = 'tanh')

        model.add(layer)
        return model