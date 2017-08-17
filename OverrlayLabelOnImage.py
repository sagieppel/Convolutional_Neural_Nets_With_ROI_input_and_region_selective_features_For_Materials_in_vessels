###################Overlay Label on image Mark label on  the image in transperent form###################################################################################
def OverLayFillLevel(ImgIn,Label,W):
    #ImageIn is the image
    #Label is the label per pixel
    # W is the relative weight in which the labels will be marked on the image
    # Return image with labels marked over it
    Img=ImgIn.copy()
    #TR=[0,255,0,255,0    ,0   ,255  ,0   ,255, 192  ,128  ,0  ,0   ,0   ,128,0,1   ,0.5]
    #TG=[0,255,0  ,255  ,0   ,255  ,255 ,0  , 192  ,0    ,128,0   ,128 ,0,0.5]
    #TB=[0,255,0  ,0    ,255 ,0    ,255 ,255, 192  ,0    ,0  ,128 ,128 ,128.5 ,0.5]
    TR = [0,1,  0.2, 0]
    TB = [0,0,  0.5, 0]
    TG = [0,0,  0.7, 1]
    R = Img[:, :, 0].copy()
    G = Img[:, :, 1].copy()
    B = Img[:, :, 2].copy()
    for i in range(1, 4):
        R[Label == i] = TR[i] * 255
        G[Label == i] = TG[i] * 255
        B[Label == i] = TB[i] * 255
    Img[:, :, 0] = Img[:, :, 0] * (1 - W) + R * W
    Img[:, :, 1] = Img[:, :, 1] * (1 - W) + G * W
    Img[:, :, 2] = Img[:, :, 2] * (1 - W) + B * W
    return Img
###################Overlay Label on image Mark label on  the image in transperent form###################################################################################
def OverLayLiquidSolid(ImgIn,Label,W):
    #ImageIn is the image
    #Label is the label per pixel
    # W is the relative weight in which the labels will be marked on the image
    # Return image with labels marked over it
    Img=ImgIn.copy()
    #TR=[0,255,0,255,0    ,0   ,255  ,0   ,255, 192  ,128  ,0  ,0   ,0   ,128,0,1   ,0.5]
    #TG=[0,255,0  ,255  ,0   ,255  ,255 ,0  , 192  ,0    ,128,0   ,128 ,0,0.5]
    #TB=[0,255,0  ,0    ,255 ,0    ,255 ,255, 192  ,0    ,0  ,128 ,128 ,128.5 ,0.5]
    TR = [0,1, 0, 0]
    TB = [0,0, 1, 0]
    TG = [0,0, 0, 1]
    R = Img[:, :, 0].copy()
    G = Img[:, :, 1].copy()
    B = Img[:, :, 2].copy()
    for i in range(1, 4):
        R[Label == i] = TR[i] * 255
        G[Label == i] = TG[i] * 255
        B[Label == i] = TB[i] * 255
    Img[:, :, 0] = Img[:, :, 0] * (1 - W) + R * W
    Img[:, :, 1] = Img[:, :, 1] * (1 - W) + G * W
    Img[:, :, 2] = Img[:, :, 2] * (1 - W) + B * W
    return Img
###################Overlay Label on image Mark label on  the image in transperent form###################################################################################
def OverLayExactPhase(ImgIn,Label,W):
    #ImageIn is the image
    #Label is the label per pixel
    # W is the relative weight in which the labels will be marked on the image
    # Return image with labels marked over it
    Img=ImgIn.copy()
    #TR=[0,255,0,255,0    ,0   ,255  ,0   ,255, 192  ,128  ,0  ,0   ,0   ,128,0,1   ,0.5]
    #TG=[0,255,0  ,255  ,0   ,255  ,255 ,0  , 192  ,0    ,128,0   ,128 ,0,0.5]
    #TB=[0,255,0  ,0    ,255 ,0    ,255 ,255, 192  ,0    ,0  ,128 ,128 ,128.5 ,0.5]
    TR = [0,1, 0, 0, 0, 1, 1, 0, 0, 0.5, 0.7, 0.3, 0.5, 1, 0.5]
    TB = [0,0, 1, 0, 1, 0, 1, 0, 0.5, 0, 0.2, 0.2, 0.7, 0.5, 0.5]
    TG = [0,0, 0, 0.5, 1, 1, 0, 1, 0.7, 0.4, 0.7, 0.2, 0, 0.25, 0.5]
    R = Img[:, :, 0].copy()
    G = Img[:, :, 1].copy()
    B = Img[:, :, 2].copy()
    for i in range(1, 14):
        R[Label == i] = TR[i] * 255
        G[Label == i] = TG[i] * 255
        B[Label == i] = TB[i] * 255
    Img[:, :, 0] = Img[:, :, 0] * (1 - W) + R * W
    Img[:, :, 1] = Img[:, :, 1] * (1 - W) + G * W
    Img[:, :, 2] = Img[:, :, 2] * (1 - W) + B * W
    return Img