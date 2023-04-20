# CSC413-Project: Multistyle Transfer

## Code
 This include the code of our project. Each file is an .ipynb and should be run on Google Colab with GPU connected. 
 
 - pretrained NST
 
   This is the pretrained NST model from [TensorFlow](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1). To run this, you can play around the code's provided data loader which will download the dataset from [UC Berkeley EECS](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) or upload yoour own. To upload your own, you should zip and upload the dataset as CSC413_test_data.zip containing content images in folder test_photo, style1 images in folder test_monet, and style2 images in folder test_vangogh
  
   Upon uploading, don't change the directory of the zip file, the code will do directory walk and unzip itself.
   
 - Model

   This is the conditional CycleGAN model. It will download the data and train itself. You can simply click Runtime -> run all to run the model
   
   Feel free to play around with the hyperparameters.
   
   The main architecture is borrowed from [Amy Jiang's implementation](https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial/notebook), and dataset is from [UC Berkeley EECS](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)

 - parse-printout

   This file is provided to extrace the loss information from console log while training the conditional CycleGAN model. Please do some preprocessing to delete irrelevant log and leaving only the two lines sating current Epoch and loss information. Then in the first cell, paste and assign variable input to the pre-processed string, and run the code
   
 - Metric

   This file contains all code for calculating the following 3 metrics: Structural Similarity (SSIM), Style Consistency Loss, and Fréchet Inception Distance (FID)
   
   To run the code, you have to upload 2 zip files: 
    - NST input output result.zip: the output of NST model
    - cycleGAN out_image.zip: the output of conditional cycleGAN model
    
    The original content images, style1 images, and style2 images should be included in NST input output result.zip. And each zip file should also include the four set of outputs: photo_to_monet, monet_to_vangogh, photo_to_vangogh, and vangogh_to_monet.
    
    To run the code, you should set the "out_path" to one of the 4 above, and uncomment either the NST or GAN part of code and choose either load_image_NST or load_image_GAN accordingly. 
    
    The code for computing FID score is borrowed form https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
