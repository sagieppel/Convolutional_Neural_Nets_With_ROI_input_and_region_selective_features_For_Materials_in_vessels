# Evaluate the perfomance of trained network by evaluating intersection over union (IOU) of the  network predcition and ground truth of the validation set
# This should work as is if you follow the instructions in the readme file (assuming you have trained network in the log dir and )
###########################################################################################################################################################
#from __future__ import print_function
import tensorflow as tf
import numpy as np



import os
import IOU
import sys
import Data_Reader
import BuildNetVgg16


logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
Label_Dir="Data_Zoo/Materials_In_Vessels/"# Annotetion for train images and validation images (assume the name of the images and annotation images are the same)
Image_Dir="Data_Zoo/Materials_In_Vessels/Test_Images_Same_Setting/"#Test images that will be used to evaluate training

model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"

#-----------------------------------------Check if models and data are availale----------------------------------------------------
if not os.path.isfile(model_path):
    print("Warning: Cant find pretrained vgg16 model for network initiation. Please download  mode from:")
    print("ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy")
    print("Or from")
    print("https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing")
    print("and place in: Model_Zoo/")

if not os.path.isdir(Image_Dir):
    print("Warning: Cant find images for evaluation in "+Image_Dir+"You can download dataset from:")
    print("https://drive.google.com/file/d/0B6njwynsu2hXRFpmY1pOV1A4SFE/view?usp=sharing")
    print("and extract in: Data_Zoo/")
#-------------------------------------------------------------------------------------------------------------------------

Batch_Size=1
NUM_CLASSES = 15 + 2 + 3 + 4 #Number of classes for all catagories combined

VesseClasses=["Background","Vessel"] #Classes predicted for vessel region prediction
PhaseClasses=["BackGround","Empty Vessel region","Filled Vessel region"]#
LiquidSolidClasses = ["BackGround", "Empty Vessel","Liquid","Solid"]
ExactPhaseClasses=["BackGround","Vessel","Liquid","Liquid Phase two","Suspension", "Emulsion","Foam","Solid","Gel","Powder","Granular","Bulk","Bulk Liquid","Solid Phase two","Vapor"]
################################################################################################################################################################################
def main(argv=None):
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")  # Dropout probability for training
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3],name="input_image")  # Input image batch first dimension image number second dimension width third dimension height fourth dimension RGB
    VesselLabel = tf.placeholder(tf.int32, shape=[None, None, None, 1],name="VesselLabel")  # Label image for vessel background prediction use as ROI input mask for the net
    #-------------------------Build Net----------------------------------------------------------------------------------------------
    Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)  # Create class instance for the net
    Net.build(image, VesselLabel, NUM_CLASSES, keep_prob) # Build net and load intial weights from vgg16 (weights before training)
    # -------------------------Data reader for validation image-----------------------------------------------------------------------------------------------------------------------------

    ValidReader = Data_Reader.Data_Reader(Image_Dir, Label_Dir, Batch_Size) # build reader that will be used to load images and labels from validation set
    sess = tf.Session()  # Start Tensorflow session
    #--------Load trained model--------------------------------------------------------------------------------------------------
    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:  # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else: # if
        print("ERROR NO TRAINED MODEL IN: " + ckpt.model_checkpoint_path+"See TRAIN.py for training")
        print("or download pretrained model from"
              " https://drive.google.com/file/d/0B6njwynsu2hXRDMxWlBUTWFZM2c/view?usp=sharing "
              "and extract in log_dir")
        sys.exit()
 #--------------------Sum of intersection from all validation images for all classes and sum of union for all images and all classes----------------------------------------------------------------------------------
    VesUn = np.float64(np.zeros(len(VesseClasses))) #Sum of union for vessel  region prediciton
    VesIn =  np.float64(np.zeros(len(VesseClasses))) #Sum of Intersection   for vessel  region prediciton
    PhaseUn =  np.float64(np.zeros(len(PhaseClasses))) #Sum of union for  for fill level prediction
    PhaseIn =  np.float64(np.zeros(len(PhaseClasses))) #Sum of Intersection  for fill level prediction
    LiqSolUn =  np.float64(np.zeros(len(LiquidSolidClasses))) #Sum of union for liquid solid prediction
    LiqSolIn =  np.float64(np.zeros(len(LiquidSolidClasses))) #Sum of Intersection for liquid solid prediction
    ExactPhaseUn =  np.float64(np.zeros(len(ExactPhaseClasses)))
    ExactPhaseIn =  np.float64(np.zeros(len(ExactPhaseClasses)))
    fim = 0
    print("Start Evaluating intersection over union for "+str(ValidReader.NumFiles)+" images")
 #===========================GO over all validation images and caclulate IOU============================================================
    while (ValidReader.itr<ValidReader.NumFiles):
        print(str(fim*100.0/ValidReader.NumFiles)+"%")
        fim+=1

#.........................................Run Predictin/inference on validation................................................................................
        Images, LabelsVessel, LabelsOnePhase, LabelsSolidLiquid, LabelsExactPhase = ValidReader.ReadNextBatchClean() # Read images and ground truth annotation
        #Predict annotation using net
        ExactPhase, LiquidSolid, OnePhase, Vessel = sess.run([Net.ExactPhasePred, Net.LiquidSolidPred, Net.PhasePred, Net.VesselPred],
                                                            feed_dict={image: Images, keep_prob: 1.0, VesselLabel:LabelsVessel})
#............................Calculate Intersection and union for prediction...............................................................

#        print("-------------------------Vessel IOU----------------------------------------")
        CIOU,CU=IOU.GetIOU(Vessel,LabelsVessel.squeeze(),len(VesseClasses),VesseClasses)
        VesIn+=CIOU*CU
        VesUn+=CU

#        print("------------------------One Phase IOU----------------------------------------")
        CIOU, CU =IOU.GetIOU(OnePhase,LabelsOnePhase.squeeze(), len(PhaseClasses), PhaseClasses)
        PhaseIn += CIOU*CU
        PhaseUn += CU

#        print("--------------------------Liquid Solid IOU-----------------------------------")
        CIOU, CU =IOU.GetIOU(LiquidSolid, LabelsSolidLiquid.squeeze() , len(LiquidSolidClasses), LiquidSolidClasses)
        LiqSolIn += CIOU*CU
        LiqSolUn += CU

#        print("----------------------All Phases  Phase IOU----------------------------------------")
        CIOU, CU =IOU.GetIOU(ExactPhase, LabelsExactPhase.squeeze(), len(ExactPhaseClasses), ExactPhaseClasses)
        ExactPhaseIn += CIOU*CU
        ExactPhaseUn += CU




#-----------------------------------------Print results--------------------------------------------------------------------------------------
    print("----------------------------------------------------------------------------------")
    print("---------------------------Mean Prediction----------------------------------------")
    print("---------------------IOU=Intersection Over Inion------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------
    print("-------------------------Vessel IOU----------------------------------------")
    for i in range(len(VesseClasses)):
        if VesUn[i]>0: print(VesseClasses[i]+"\t"+str(VesIn[i]/VesUn[i]))
    print("------------------------One Phase IOU----------------------------------------")
    for i in range(len(PhaseClasses)):
        if PhaseUn[i] > 0: print(PhaseClasses[i]+"\t"+str(PhaseIn[i]/PhaseUn[i]))
    print("--------------------------Liquid Solid IOU-----------------------------------")
    for i in range(len(LiquidSolidClasses)):
        if LiqSolUn[i] > 0: print(LiquidSolidClasses[i]+"\t"+str(LiqSolIn[i]/LiqSolUn[i]))
    print("----------------------All Phases  Phase IOU----------------------------------------")
    for i in range(len(ExactPhaseClasses)):
        if ExactPhaseUn[i]> 0: print(ExactPhaseClasses[i]+"\t"+str(ExactPhaseIn[i]/ExactPhaseUn[i]))

##################################################################################################################################################
main()#Run script
print("Finished")