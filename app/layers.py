#import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

#we need to load the custom model as custom object inject into model trained we just saved

# Siamese L1 Distance class
class L1Dist(Layer): # create out own layer, inherit from Layer class
    
    # Init method - inheritance，kwargs=key word arguments(dict)
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding) 
    #return a tensor(same number as input), each element is absolute value of corresponding element in input