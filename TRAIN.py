# Train fully conbolional neural net with valve filters for pixelwise annotation of materials in transperent vessel

# Output trained network should appear in the /log folder (or in the folder pointer by log_dir)
# By defualt this should work as is, if you follow the intsructions provide in the readme file
# (i.e downloaded the materials in vessel the data set to the Data_Zoo folder)
##########################################################################################################################################################################
import tensorflow as tf
import numpy as np
import Data_Reader
import BuildNetVgg16
import TensorflowUtils
import os
Train_Image_Dir="Data_Zoo/Materials_In_Vessels/Train_Images/" # Images and labels for training
if not os.path.exists("Data_Zoo/"): os.makedirs("Data_Zoo/")
Label_Dir="Data_Zoo/Materials_In_Vessels/"# Annotetion for train images and validation images (assume the name of the images and annotation images are the same)
Valid_Image_Dir="Data_Zoo/Materials_In_Vessels/Test_Images_All/"# Validation images that will be used to evaluate training
logs_dir= "logs/"# "path to logs directory where trained model and information will be stored"
if not os.path.exists(logs_dir): os.makedirs(logs_dir)
model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"
learning_rate=1e-5 #Learning rate for Adam Optimizer
if not os.path.exists("Model_Zoo/"): os.makedirs("Model_Zoo/")
#-----------------------------------------Check if models and data are availale----------------------------------------------------
if not os.path.isfile(model_path):
    print("Warning: Cant find pretrained vgg16 model for network initiation. Please download model from:")
    print("ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy")
    print("Or from")
    print("https://drive.google.com/file/d/0B6njwynsu2hXZWcwX0FKTGJKRWs/view?usp=sharing")
    print("and place in: Model_Zoo/")

if not os.path.isdir(Train_Image_Dir) or not  os.path.isdir(Label_Dir):
    print("Warning: Cant find dataset for training. Please download from:")
    print("https://drive.google.com/file/d/0B6njwynsu2hXQzZTWTRBLWRUT0U/view?usp=sharing")
    print("and extract in: Data_Zoo/")
TensorflowUtils.maybe_download_and_extract(model_path.split('/')[0], "ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy") #If not exist try to download pretrained vgg16 net for network initiation
#This can be download manualy from:"https://drive.google.com/drive/folders/0B6njwynsu2hXcDYwb1hxMW9HMEU" or from ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy
# and placed in the /Model_Zoo folder in the code dir
#-----------------------------------------------------------------------------------------------------


TrainLossTxtFile=logs_dir+"TrainLoss.txt" #Where train losses will be writen
ValidLossTxtFile=logs_dir+"ValidationLoss.txt"# Where validation losses will be writen


Batch_Size=2 # Number of files per training iteration


Weight_Loss_Rate=5e-4# Weight for the weight decay loss function
MAX_ITERATION = int(100010) # Max  number of training iteration
NUM_CLASSES = 15+2+3+4#Number of class for fine grain +number of class for solid liquid+Number of class for empty none empty +Number of class for vessel background
######################################Solver for model   training#####################################################################################################################
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

################################################################################################################################################################################
################################################################################################################################################################################
def main(argv=None):
    tf.reset_default_graph()
    keep_prob= tf.placeholder(tf.float32, name="keep_probabilty") #Dropout probability
#.........................Placeholders for input image and labels...........................................................................................
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image") #Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB
    VesselLabel = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="VesselLabel")  # Label image for vessel background prediction
    PhaseLabel = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="PhaseLabel")#Label image for Vessel Full  and background prediction
    LiquidSolidLabel = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="LiquidSolidLabel")  # Label image  for liquid solid  vessel  background prediction
    ExactPhaseLabel = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="ExactPhaseLabel")  # Label image for fine grain phase prediction (liquid solid powder foam
#.........................Build FCN Net...............................................................................................
    Net =  BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path) #Create class for the network
    Net.build(image, VesselLabel,NUM_CLASSES,keep_prob)# Create the net and load intial weights
#......................................Get loss functions for neural net work  one loss function for each set of label....................................................................................................
    VesselLoss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(VesselLabel, squeeze_dims=[3]), logits=Net.VesselProb,name="VesselLoss")))  # Define loss function for training
    PhaseLoss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(PhaseLabel, squeeze_dims=[3]), logits=Net.PhaseProb,name="PhaseLoss")))  # Define loss function for training
    LiquidSolidLoss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(LiquidSolidLabel, squeeze_dims=[3]), logits=Net.LiquidSolidProb,name="LiquidSolidLoss")))  # Define loss function for training
    ExactPhaseLoss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(ExactPhaseLabel, squeeze_dims=[3]), logits=Net.ExactPhaseProb,name="PhaseLabel")))  # Define loss function for training
    WeightDecayLoss=Net.SumWeights*Weight_Loss_Rate #Weight decay loss
    TotalLoss=VesselLoss+PhaseLoss+LiquidSolidLoss+ExactPhaseLoss+WeightDecayLoss# Loss  is  the  sum of loss for all categories
#....................................Create solver for the net............................................................................................
    trainable_var = tf.trainable_variables()
    train_op = train(TotalLoss, trainable_var)
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
    TrainReader = Data_Reader.Data_Reader(Train_Image_Dir, Label_Dir,Batch_Size) #Reader for training data
    ValidReader = Data_Reader.Data_Reader(Valid_Image_Dir, Label_Dir, Batch_Size) # Reader for validation data
    sess = tf.Session() #Start Tensorflow session
# -------------load trained model if exist-----------------------------------------------------------------

    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if train model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

#---------------------------Start Training: Create loss files for saving loss during traing ----------------------------------------------------------------------------------------------------------

    f = open(TrainLossTxtFile, "w")
    f.write("Iteration\tTotal_Loss\tVessel Loss\tFill_Level_Loss%\tLiquid_Solid_Loss\tExact_Phases_loss\t Learning Rate="+str(learning_rate))
    f.close()
    f = open(ValidLossTxtFile, "w")
    f.write("Iteration\tTotal_Loss\tVessel Loss\tFill_Level_Loss%\tLiquid_Solid_Loss\tExact_Phases_loss\t Learning Rate=" + str(learning_rate))
    f.close()
#..............Start Training loop: Main Training....................................................................
    for itr in range(MAX_ITERATION):
        Images,LabelsVessel,LabelsOnePhase,LabelsSolidLiquid,LabelsExactPhase=TrainReader.ReadAndAugmentNextBatch() # Load  augmeted images and ground true labels for training
        feed_dict = {image: Images,VesselLabel:LabelsVessel, PhaseLabel: LabelsOnePhase,LiquidSolidLabel:LabelsSolidLiquid,ExactPhaseLabel:LabelsExactPhase, keep_prob: 0.5}
        sess.run(train_op, feed_dict=feed_dict) # Train one cycle
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
        if itr % 5000 == 0 and itr>0: saver.save(sess, logs_dir + "model.ckpt", itr)
#......................Write and display train loss..........................................................................
        if itr % 10==0:
            # Calculate train loss
            Tot_Loss,Ves_Loss,Phase_Loss,LiquidSolid_Loss,ExactPhase_Loss= sess.run([TotalLoss,VesselLoss, PhaseLoss,LiquidSolidLoss,ExactPhaseLoss], feed_dict=feed_dict)
            print("Step: %d,  Total_loss:%g,  Vessel_Loss:%g,  OnePhases_Loss:%g,  LiquidSolid_Loss:%g,  ExactPhase_Loss:%g," % (itr, Tot_Loss,Ves_Loss,Phase_Loss,LiquidSolid_Loss,ExactPhase_Loss))
            #Write train loss to file
            with open(TrainLossTxtFile, "a") as f:
                f.write("\n"+str(itr)+"\t"+str(Tot_Loss)+"\t"+str(Ves_Loss)+"\t"+str(Phase_Loss)+"\t"+str(LiquidSolid_Loss)+"\t"+str(ExactPhase_Loss))
                f.close()

#......................Write and display Validation Set Loss by running loss on all validation images.....................................................................
        if itr % 500 == 0:

            SumTotalLoss=np.float64(0.0)
            SumVesselLoss = np.float64(0.0)
            SumOnePhassLoss = np.float64(0.0)
            SumLiquidSolidLoss= np.float64(0.0)
            SumExactPhase_Loss= np.float64(0.0)

            NBatches=np.int(np.ceil(ValidReader.NumFiles/ValidReader.BatchSize))
            for i in range(NBatches):# Go over all validation image
                Images, LabelsVessel, LabelsOnePhase, LabelsSolidLiquid, LabelsExactPhase = ValidReader.ReadNextBatchClean() # load validation image and ground true labels
                feed_dict = {image: Images, VesselLabel: LabelsVessel, PhaseLabel: LabelsOnePhase,LiquidSolidLabel: LabelsSolidLiquid, ExactPhaseLabel: LabelsExactPhase,keep_prob: 1}
                # Calculate loss for all labels set
                Tot_Loss, Ves_Loss, Phase_Loss, LiquidSolid_Loss, ExactPhase_Loss = sess.run([TotalLoss, VesselLoss, PhaseLoss, LiquidSolidLoss, ExactPhaseLoss], feed_dict=feed_dict)

                SumTotalLoss+=Tot_Loss
                SumVesselLoss+=Ves_Loss
                SumOnePhassLoss+=Phase_Loss
                SumLiquidSolidLoss+=LiquidSolid_Loss
                SumExactPhase_Loss+=ExactPhase_Loss
                NBatches+=1

            SumTotalLoss/=NBatches
            SumVesselLoss /= NBatches
            SumOnePhassLoss /= NBatches
            SumLiquidSolidLoss/=NBatches
            SumExactPhase_Loss/= NBatches
            print("Validation Total_loss:%g,  Vessel_Loss:%g,  OnePhases_Loss:%g,  LiquidSolid_Loss:%g,  ExactPhase_Loss:%g," % (SumTotalLoss, SumVesselLoss, SumOnePhassLoss, SumLiquidSolidLoss, SumExactPhase_Loss))
            with open(ValidLossTxtFile, "a") as f:

                f.write("\n" + str(itr) + "\t" + str(SumTotalLoss) + "\t" + str(SumVesselLoss) + "\t" + str(SumOnePhassLoss) + "\t" + str(SumLiquidSolidLoss) + "\t" + str(SumExactPhase_Loss))
                f.close()


##################################################################################################################################################
main()#Run script
print("Finished")
