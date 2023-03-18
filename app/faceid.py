#import kivy dependencies
#kivy is cross-platform python frameowkr for building app
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

#kivy ux components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

#other kivy components
from kivy.clock import Clock

# we use this to convert our image from opencv webcam to a texture for further comparison
from kivy.graphics.texture import Texture
from kivy.logger import Logger


import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


class CameraApp(App):
  # this function inherit from App
  def build(self):
    # size_hint, 1 for full width, 0.8 for the height(image take 80% of height)
    self.webcam = Image(size_hint=(1,.8))
    self.button = Button(text='Verify',on_press=self.verify, size_hint=(1,.1))
    self.verificationLabel = Label(text="Verification Uninitiated", size_hint=(1,.1))
    layout = BoxLayout(orientation='vertical')
    layout.add_widget(self.webcam)
    layout.add_widget(self.verificationLabel)
    layout.add_widget(self.button)

    self.modal = tf.keras.models.load_model('siamesemodelv2.h5',custom_objects={'L1Dist': L1Dist})

    self.capture = cv2.VideoCapture(0)

    # we run the funciton update by that interval
    Clock.schedule_interval(self.update, 1/33)
    return layout
  

  def update(self, *args): 
    ret, frame = self.capture.read()
    frame = frame[120:120+250,200:200+250, :]

    #flip the image horizontally
    buffer = cv2.flip(frame,0).tostring()
    # size=height,width
    image_texture = Texture.create(size=(frame.shape[1],frame.shape[0]), colorfmt='bgr')
    #take the image and convert it into a texture
    image_texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')

    #render the image to the webcam object we put in the boxlayout
    self.webcam.texture = image_texture



  #preprocess the image
  def preprocess(self, file_path): #like this file path data\\anchor\\5d3a7c8e-bf4b-11ed-a755-c403a8279614.jpg
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image(decode jpg file)，to array of pixel by pixel(R,G,B) this is called TENSOR
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image(from 0-255 color) to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img
  


  def verify(self, *args):
    detection_threshold = 0.90
    verification_threshold = 0.70
    savePath = os.path.join('application_data', 'input_image', 'input_image.jpg')
    ret,frame = self.capture.read()
    frame = frame[120:120+250,200:200+250, :]
    cv2.imwrite(savePath,frame)
    # Build results array
    results = []
    # loop over all the images inside the verification_images folder
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
        
        # Make Predictions 
        result = self.modal.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    verified = verification > verification_threshold
    #set verification label
    self.verificationLabel.text = "Verified " if verified else 'Unverified'

    Logger.info(results)
    Logger.info(verification)
    return results, verified





# this will run only run this file
#if use main() directly, it will also run when imported
if __name__ == '__main__':
    CameraApp().run() #function inherited from App class
