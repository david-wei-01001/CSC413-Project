# CSC413-Project: Monetizing Arts

## Code
 This include the code of our project. Each file is an .ipynb and should be run on Google Colab with GPU connected. 
 
 - pretrained NST
 
   This is the pretrained NST model from [TensorFlow](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1). To run this, you can play around the code's provided data loader which will download the dataset from [UC Berkeley EECS](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) or upload yoour own. To upload your own, you should zip and upload the dataset as CSC413_test_data.zip containing content images in folder test_photo, style1 images in folder test_monet, and style2 images in folder test_vangogh
  
   Upon uploading, don't change the directory of the zip file, the code will do directory walk and unzip itself.
   
 - Model

   This is the conditional CycleGAN model. It will download the data and train itself. You can simply click Runtime -> run all to run the model
   
   Feel free to play around with the hyperparameters.
   
   The main architecture is borrowed from [Amy Jiang's implementation](https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial/notebook), and dataset is from [UC Berkeley EECS](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)
