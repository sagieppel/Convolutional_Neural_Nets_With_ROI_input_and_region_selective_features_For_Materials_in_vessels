# Take Fine grain labels classe and generate course grain labels
# Note the data set already contain coarse grained labels so this is not needed
import numpy as np
import os
import scipy.misc as misc

#Input fine  grain labels dir
LabelDir="/home/sagi/TENSORFLOW/DATASETS/Materials_In_Vessels_Dataset_Final/LabelsAll/" #Fine grain label input dir for all
# output Coarse grain annotation dir
OutLabelDir="/home/sagi/TENSORFLOW/DATASETS/Materials_In_Vessels_Dataset_Final/"
VesselDir=OutLabelDir+"/VesselLabels/"
FillLevelDir=OutLabelDir+"/FillLevelLabels/"
LiquidSolidDir=OutLabelDir+"/LiquidSolidLabels/"
AllPhasesDir=OutLabelDir+"/AllPhasesLabels/"
if not os.path.exists(VesselDir): os.mkdir(VesselDir)
if not os.path.exists(FillLevelDir): os.mkdir(FillLevelDir)
if not os.path.exists(LiquidSolidDir): os.mkdir(LiquidSolidDir)
if not os.path.exists(AllPhasesDir): os.mkdir(AllPhasesDir)
LabelFiles=[]
LabelFiles += [each for each in os.listdir(LabelDir) if each.endswith('.png')]
for itr in range(len(LabelFiles)):
    print(itr)
    Label = misc.imread(LabelDir+LabelFiles[itr])
    #if Label==None: print("Fail to read: "+LabelFiles[itr])

    OutLabel=np.zeros(Label.shape)
    OutLabel[Label>0]=1
    misc.imsave(VesselDir + LabelFiles[itr], OutLabel.astype(np.uint8))
    OutLabel[Label > 1] = 2
    OutLabel[Label == 14] = 1  # vapor-empty
    misc.imsave(FillLevelDir+LabelFiles[itr], OutLabel.astype(np.uint8))

    OutLabel[Label > 1]=2 #Liquid
    OutLabel[Label >6]=3#solid
    #OutLabel[Label == 12] = 2  # liquid
    OutLabel[Label == 14] = 1  # vapor-empty
    misc.imsave(LiquidSolidDir + LabelFiles[itr], OutLabel.astype(np.uint8))
    misc.imsave(AllPhasesDir + LabelFiles[itr], Label.astype(np.uint8))
#Fine Grain Classes
#All Phases Classes Labels=[0"Empty",1"Vessel",2"Liquid",3"Liquid Phase two",4"Suspension", 5"Emulsion",6"Foam",7"Solid",8"Gel","9Powder",10"Granular",11"Bulk",12"Bulk Liquid",13"Solid Phase",14"Vapor"]
#Coarse Grain Classes
#Vesse Classes Labels=[0"Background",1"Vessel"]
#Phase Classes Labels =["0BackGround",1"Empty Vessel",2"Filled"]
#Liquid Solid Classes = [0"BackGround", 1"Empty Vessel",2"Liquid",3"Solid"]
