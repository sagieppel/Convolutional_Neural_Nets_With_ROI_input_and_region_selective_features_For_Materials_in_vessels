# Run prediction and genertae pixelwise annotation for every pixels in the image based on image and ROI mask.
# Output saved as label images, and label image overlay on the original image
# By defualt this should work, as is if you follow the intsructions provide in the readme file
# 1) Make sure you you have trained model in logs_dir (See Train.py for creating trained model)
# 2) Set the Image_Dir to the folder where the input image for prediction are located
# 3) Set Pred_Dir the folder where you want the output annotated images to be save
# 4) Set the Label dir to the perent label folder to the materials in vessel data set main dir (you need the vessel label to be used as ROI input)
# (Currently the ROI map is read from the materials in vessel data set Vessel region label)
#--------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import sys
import BuildNetVgg16
import TensorflowUtils
import os
import Data_Reader
import OverrlayLabelOnImage as Overlay
logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
Label_Dir="Data_Zoo/Materials_In_Vessels/"# Annotetion for train images and validation images (assume the name of the images and annotation images are the same)
Image_Dir="Data_Zoo/Materials_In_Vessels/Test_Images_All"# Test image for annotation prediction
w=0.7# weight of overlay on image
#Data_Zoo/Materials_In_Vessels/Test_Images/"# Images for testing network
Pred_Dir="Output_Prediction/" # Library where the output prediction will be written
model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
NameEnd="" # Add this string to the ending of the file name optional

#-----------------------------------------Check if models and data are availale----------------------------------------------------
if not os.path.isfile(model_path):
    print("Warning: Cant find pretrained vgg16 model for network initiation. Please download  mode from:")
    print("ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy")
    print("Or from")
    print("https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing")
    print("and place in: Model_Zoo/")

if not os.path.isdir(Image_Dir):
    print("Warning: Cant find images for interference. You can downolad test images from:")
    print("https://drive.google.com/file/d/0B6njwynsu2hXQzZTWTRBLWRUT0U/view?usp=sharing")
    print("and extract in: Data_Zoo/")
TensorflowUtils.maybe_download_and_extract(model_path.split('/')[0], "ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy") #If not exist try to download pretrained vgg16 net for network initiation
#-------------------------------------------------------------------------------------------------------------------------
NUM_CLASSES = 15+2+3+4


################################################################################################################################################################################
def main(argv=None):
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")  # Dropout probability
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                           name="input_image")  # Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB
    VesselLabel = tf.placeholder(tf.int32, shape=[None, None, None, 1],
                                 name="VesselLabel")  # Label image for vessel background prediction
    # -------------------------Build Net----------------------------------------------------------------------------------------------
    Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)  # Create class instance for the net
    Net.build(image, VesselLabel, NUM_CLASSES, keep_prob)  # Build net and load intial weights (weights before training)
    # -------------------------Data reader for validation/testing images-----------------------------------------------------------------------------------------------------------------------------
    ValidReader = Data_Reader.Data_Reader(Image_Dir, Label_Dir, 1)






    #-------------------------Load Trained model if you dont have trained model see: Train.py-----------------------------------------------------------------------------------------------------------------------------

    sess = tf.Session() #Start Tensorflow session

    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        print("ERROR NO TRAINED MODEL IN: "+ckpt.model_checkpoint_path+" See Train.py for creating train network ")
        print("or download pretrained model from"
              "https://drive.google.com/file/d/0B6njwynsu2hXRFpmY1pOV1A4SFE/view?usp=sharing"
              "and extract in log_dir")
        sys.exit()


#--------------------Create output directories for predicted label, one folder for each granulairy of label prediciton---------------------------------------------------------------------------------------------------------------------------------------------

    if not os.path.exists(Pred_Dir): os.makedirs(Pred_Dir)
    if not os.path.exists(Pred_Dir+"/OverLay"): os.makedirs(Pred_Dir+"/OverLay")
    if not os.path.exists(Pred_Dir + "/OverLay/Vessel/"): os.makedirs(Pred_Dir + "/OverLay/Vessel/")
    if not os.path.exists(Pred_Dir + "/OverLay/OnePhase/"): os.makedirs(Pred_Dir + "/OverLay/OnePhase/")
    if not os.path.exists(Pred_Dir + "/OverLay/LiquiSolid/"): os.makedirs(Pred_Dir + "/OverLay/LiquiSolid/")
    if not os.path.exists(Pred_Dir + "/OverLay/ExactPhase/"): os.makedirs(Pred_Dir + "/OverLay/ExactPhase/")
    if not os.path.exists(Pred_Dir + "/Label"): os.makedirs(Pred_Dir + "/Label")
    if not os.path.exists(Pred_Dir + "/Label/Vessel/"): os.makedirs(Pred_Dir + "/Label/Vessel/")
    if not os.path.exists(Pred_Dir + "/Label/OnePhase/"): os.makedirs(Pred_Dir + "/Label/OnePhase/")
    if not os.path.exists(Pred_Dir + "/Label/LiquiSolid/"): os.makedirs(Pred_Dir + "/Label/LiquiSolid/")
    if not os.path.exists(Pred_Dir + "/Label/ExactPhase/"): os.makedirs(Pred_Dir + "/Label/ExactPhase/")
    if not os.path.exists(Pred_Dir + "/AllPredicitionsDisplayed/"): os.makedirs(Pred_Dir + "/AllPredicitionsDisplayed/")
    
    print("Running Predictions:")
    print("Saving output to:" + Pred_Dir)
 #----------------------Go over all images and predict semantic segmentation in various of classes-------------------------------------------------------------
    fim = 0
    print("Start Predicting " + str(ValidReader.NumFiles) + " images")
    while (ValidReader.itr < ValidReader.NumFiles):
        print(str(fim * 100.0 / ValidReader.NumFiles) + "%")
        fim += 1

        # ..................................Load image.......................................................................................
        FileName=ValidReader.OrderedFiles[ValidReader.itr]
        Images, LabelsVessel, LabelsOnePhase, LabelsSolidLiquid, LabelsExactPhase = ValidReader.ReadNextBatchClean()  # Read images and ground truth annotation
        # Predict annotation using net
        ExactPhase, LiquidSolid, OnePhase, Vessel = sess.run(
            [Net.ExactPhasePred, Net.LiquidSolidPred, Net.PhasePred, Net.VesselPred],
            feed_dict={image: Images, keep_prob: 1.0, VesselLabel: LabelsVessel})
        #------------------------Save predicted labels overlay on images---------------------------------------------------------------------------------------------
        misc.imsave(Pred_Dir + "/OverLay/Vessel/"+ FileName+NameEnd  , Overlay.OverLayLiquidSolid(Images[0],Vessel[0], w))
        misc.imsave(Pred_Dir + "/Label/OnePhase/" + FileName + NameEnd, OnePhase[0])
        misc.imsave(Pred_Dir + "/OverLay/OnePhase/" + FileName + NameEnd,Overlay.OverLayFillLevel(Images[0], OnePhase[0], w))
        misc.imsave(Pred_Dir + "/Label/LiquiSolid/" + FileName + NameEnd, LiquidSolid[0])
        misc.imsave(Pred_Dir + "/OverLay/LiquiSolid/" + FileName + NameEnd,Overlay.OverLayLiquidSolid(Images[0], LiquidSolid[0], w))
        misc.imsave(Pred_Dir + "/Label/ExactPhase/" + FileName + NameEnd,  ExactPhase[0])
        misc.imsave(Pred_Dir + "/OverLay/ExactPhase/" + FileName + NameEnd,Overlay.OverLayExactPhase(Images[0], ExactPhase[0], w))
        misc.imsave(Pred_Dir + "/AllPredicitionsDisplayed/" + FileName+ NameEnd,np.concatenate((Images[0], Overlay.OverLayLiquidSolid(Images[0],Vessel[0],w),Overlay.OverLayFillLevel(Images[0], OnePhase[0], w),Overlay.OverLayLiquidSolid(Images[0], LiquidSolid[0], w), Overlay.OverLayExactPhase(Images[0], ExactPhase[0], w)), axis=1))
##################################################################################################################################################
main()#Run script
print("Finished")