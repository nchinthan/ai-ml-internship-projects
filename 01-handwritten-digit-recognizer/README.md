\# Handwritten Digit Recognizer (MNIST)



A Deep Learning project that uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify handwritten digits (0–9) from the classic MNIST dataset.



\## Project Overview



\- \*\*Dataset:\*\* MNIST (60,000 training images, 10,000 test images)

\- \*\*Model Architecture:\*\* Convolutional Neural Network (CNN)

\- \*\*Framework:\*\* TensorFlow / Keras

\- \*\*Goal:\*\* Achieve high accuracy (>98%) on 28x28 grayscale handwritten digit recognition.



\## Model Architecture



1\. \*\*Conv2D Layer:\*\* 32 filters, 3x3 kernel, ReLU activation

2\. \*\*MaxPooling2D Layer:\*\* 2x2 pool size

3\. \*\*Conv2D Layer:\*\* 64 filters, 3x3 kernel, ReLU activation

4\. \*\*MaxPooling2D Layer:\*\* 2x2 pool size

5\. \*\*Flatten Layer:\*\* Converts 2D feature maps to 1D vector

6\. \*\*Dense Layer:\*\* 128 units, ReLU activation

7\. \*\*Dropout Layer:\*\* Rate 0.5 (prevents overfitting)

8\. \*\*Output Layer:\*\* 10 units, Softmax activation



\## How to Run



\### 1. Install Dependencies

```bash

pip install tensorflow numpy matplotlib pillow

