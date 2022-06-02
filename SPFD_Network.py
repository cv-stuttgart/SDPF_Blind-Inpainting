from tensorflow import keras

from numpy import floor, ceil

from conv2d_LC_layer import Conv2D_LC

class MixedLayer(keras.layers.Layer):
  def __init__(self, outputchannels, kernelsize, activation):
    super(MixedLayer, self).__init__()
    self.outputchannelssparse = int(floor(outputchannels / 2.))
    self.outputchannelsconv = int(ceil(outputchannels / 2.))

    self.sparselayer = Conv2D_LC(num_filters = self.outputchannelssparse,
                               sample_size = 3,
                               kernel_size = (kernelsize,kernelsize),
                               activation  = activation,
                               padding     = 'same',
                               data_format = 'channels_last')
    self.convlayer = keras.layers.Conv2D(self.outputchannelsconv, (kernelsize,kernelsize), padding='same', activation=activation)

  def call(self, inputs):
    inputssparse = self.sparselayer(inputs)
    inputsconv = self.convlayer(inputs)
    return keras.layers.concatenate([inputssparse, inputsconv], axis=-1)
    #tf.matmul(inputs, self.kernel)

  def get_sparselayer(self):
    return self.sparselayer

  def get_convlayer(self):
    return self.convlayer


class SPFDModel(keras.Model):
    #def __init__(self, grayscale = False):
    def __init__(self, lays = [('conv', 5, 64, 'relu', 0), ('conv', 5, 64, 'relu', 0), ('conv', 1, 48, 'relu', 0), ('conv', 5, 32, 'relu', 0), ('conv', 5, 32, 'relu', 0), ('conv', 5, 4, 'relu', 0)]):
        super(SPFDModel,self).__init__()

        print("Constructing SPFD Model")

        self.lay1 = self._return_layer(lays[0])
        self.lay2 = self._return_layer(lays[1])
        self.lay3 = self._return_layer(lays[2])
        self.lay4 = self._return_layer(lays[3])
        self.lay5 = self._return_layer(lays[4])
        self.lay6 = self._return_layer(lays[5])

        print("\tdone.\n")


    def _return_layer(self, layspec):
        (layertype, kernelsize, outputchannels, activation, alpha) = layspec

        if layertype == 'conv':
            lay = keras.layers.Conv2D(outputchannels, (kernelsize,kernelsize), padding='same', activation=activation)
            print("\tadding Conv   layer with outchannels=%s, kernelsize %s, activation=%s." % (str(outputchannels), str(kernelsize), activation))

        elif layertype == 'sdpf':
            lay = Conv2D_LC(num_filters = outputchannels,
                               sample_size = 3,
                               kernel_size = (kernelsize,kernelsize),
                               activation  = activation,
                               padding     = 'same',
                               data_format = 'channels_last')
            print("\tadding sparse directional parseval frame layer with outchannels=%s, kernelsize %s, activation=%s." % (str(outputchannels), str(kernelsize), activation))

        else:
            raise ValueError("Invalid layertype, please use 'conv' or 'sdpf'")

        return lay


    def call(self, inputs):

        result = self.lay1(inputs)
        result = self.lay2(result)
        result = self.lay3(result)
        result = self.lay4(result)
        result = self.lay5(result)
        result = self.lay6(result)

        return result

        return result

