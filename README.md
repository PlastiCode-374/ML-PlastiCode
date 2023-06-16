# MachineLearning
Machine learning is a branch of artificial intelligence that enables computers to learn from data and perform tasks without being explicitly programmed. In our application, we use machine learning models with TensorFlow lite and tensorflow to detect the type of plastic waste from the images uploaded by the users. By using machine learning with image classification, we can provide accurate and relevant information and suggestions for each type of plastic waste.
To create a Machine Learning model in this project there are several parameters, tools, and libraries that we use, including the following:

## CNN Model
CNN (Convolutional Neural Network), which is a type of deep learning model commonly used in computer vision tasks. CNNs are specifically designed to process grid-like data, such as images, by applying convolutional layers that detect local patterns and hierarchical representations.
Convolutional Neural Networks have revolutionized the field of computer vision and achieved remarkable success in various tasks, including image classification, object detection, image segmentation, and more. CNN models are composed of multiple layers, including convolutional layers, pooling layers, and fully connected layers. To train our CNN model, we need a large amount of data.
1. We collect data sets of seven types of plastic waste based on the resin identification code (RIC). We obtain the data sets from various sources, such as Kaggle, Google Images, and take pictures directly.
2. Before inputting the data sets to the CNN model, we need to do some pre-processing techniques, such as resizing, cropping, augmenting, and normalizing the images and do some sorting of images.

## Google Collab
Google Colab, short for Google Colaboratory, is an online platform provided by Google that allows users to write and execute Python code in a browser-based environment. It offers a Jupyter Notebook interface that enables users to create and share documents containing live code, equations, visualizations, and explanatory text.
We use Google Colab to work together and train our CNN model. After training our CNN model, we evaluate its performance using metrics such as accuracy, precision, recall, and F1-score. We also test our model using new images that are not in the training or validation data sets. Our model achieves an accuracy of more than 85%, which means it can classify most of the images correctly.
