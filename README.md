<h1 align="center">Facial Recognition App </h1>

---

<p align="center"> Image authentication app by Python and TensorFlow
    <br> 
</p>

## 🏁 Demo
The following video is a demonstration of this app.

https://user-images.githubusercontent.com/97903569/227969632-21153a49-0cb4-4b40-a824-9b57e3e700b7.mp4

## 🎈 Features <a name="usage"></a>

- Using `Tensorflow` and `Keras` for Deep Learning
- Using `OpenCV` for image capture and processing, so that we can verify the image in real-time
- Using `Kivy` App for image verification
- Training model replicates what is shown in the paper titled Siamese Neural Networks for One-shot Image Recognition
- Model is implemented using the `Convolutional Neural Network`

## ⛏️ Usage <a name = "built_using"></a>
1. Install the `Jupyter Notebook` to open the Facial verification AI.ipynb file.
2. Follow the instructions in that file to capture verification images and train the model. So it can recognize and verify your face.
3. You can get false samples from this [Link](http://vis-www.cs.umass.edu/lfw/)
4. Copy the `siamesemodelv2.h5` file from the base directory to the app directory
5. Run faceid.py to start the application. The application will use the model you just trained to verify your images  
6. Please check this [paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) for Siamese Neural Networks explanation


Important!!  
Don't forget to change the number in `cv2.VideoCapture()` to match the camera for image capture. Please see the Jupyter file for more details.
