#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Arjun Gupta
M.Eng. Student in Robotics,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

# import tensorflow as tf
import tensorflow as tf
# tf.disable_v2_behavior()
import cv2
import os
import sys
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
# from Network.Network import CIFAR10Model
# from Network.Network import DenseNet
from Network.Network import ResNet
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.png'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.png')

    return ImageSize, DataPath
    
def ReadImages(ImageSize, DataPath):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    ImageName = DataPath
    
    I1 = cv2.imread(ImageName)
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    # I1S = iu.StandardizeInputs(np.float32(I1))

    # I1Combined = np.expand_dims(I1S, axis=0)

    # return I1Combined, I1
    # lower_limit = 0
    # upper_limit = 1
    # I1 = (upper_limit - lower_limit) *(I1 - np.max(I1))/(np.max(I1) \
    #                                                 - np.min(I1)) - upper_limit
    I1 = (I1-np.mean(I1))/255
    I1Combined = np.expand_dims(I1, axis=0)

    return I1Combined, I1

                

def TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    Length = ImageSize[0]
    # Predict output with forward pass, MiniBatchSize for Test is 1
    _, prSoftMaxS = ResNet(ImgPH, ImageSize, 1)

    # Setup Saver
    Saver = tf.train.Saver()

    
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        OutSaveT = open(LabelsPathPred, 'w')
        print(np.size(DataPath))
        for count in tqdm(range(np.size(DataPath))):            
            DataPathNow = DataPath[count]
            Img, ImgOrg = ReadImages(ImageSize, DataPathNow)
            FeedDict = {ImgPH: Img}
            PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))
            OutSaveT.write(str(PredT)+'\n')
            
        OutSaveT.close()

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    num_classes = 10

    # Get the confusion matrix using sklearn.
    print('length = '+ str(len(LabelsPred)))
    print('length = '+ str(len(LabelsTrue)))
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')

    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.imshow(cm)
    plt.savefig('./plots/test_resnet/colorbar.png')
    plt.show()

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """
    accTestOverEpochs=np.array([0,0])

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumEpochs', type=int, default=30, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/vishnuu/Documents/Arjun/CIFAR10/Test/', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Args = Parser.parse_args()
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath
    NumEpochs = Args.NumEpochs
    start = time.time()

    for epoch in range(NumEpochs):
        # Parse Command Line arguments
        tf.reset_default_graph()

        ModelPath = '/home/vishnuu/Documents/Arjun/Checkpoints_resnet/'+str(epoch)+'model.ckpt'
        # print(ModelPath)
        # BasePath = Args.BasePath
        # LabelsPath = Args.LabelsPath

        # Setup all needed parameters including file reading
        ImageSize, DataPath = SetupAll(BasePath)

        # Define PlaceHolder variables for Input and Predicted output
        ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
        LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

        TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred)

        # Plot Confusion Matrix
        LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
        accuracy=Accuracy(LabelsTrue, LabelsPred)
        print(accuracy)
        accTestOverEpochs=np.vstack((accTestOverEpochs,[epoch,accuracy]))
        plt.xlim(0,NumEpochs)
        plt.ylim(0,100)
        plt.xlabel('Epoch')
        plt.ylabel('Test accuracy')
        plt.subplots_adjust(hspace=0.6,wspace=0.3)
        plt.plot(accTestOverEpochs[:,0],accTestOverEpochs[:,1])
        plt.savefig('./plots/test_resnet/Epochs'+str(epoch)+'.png')
        plt.close()
        # ConfusionMatrix(LabelsTrue, LabelsPred)
    end = time.time()
    infer_time = (end-start)/10000
    print('inference time:',infer_time)
    ConfusionMatrix(LabelsTrue, LabelsPred)


     
if __name__ == '__main__':
    main()
 