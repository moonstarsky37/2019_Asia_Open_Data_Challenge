import time, os, sys
from util.ziyu_model import *
from util.train import *
from util.data_parser import *
from util.evaluate import *

class ArgumentForLoss():
    def __init__(self, model_dest, blocks, block_conv, block_channels, channels,
                loss, batch_size, epochs):
        self.model_dest = model_dest
        self.blocks = blocks
        self.block_conv = block_conv
        self.block_channels = block_channels
        self.channels = channels
        self.loss = loss
        self.box_size = 128
        self.batch_size = 30
        self.epochs = 1#300

if __name__ == '__main__':
    ################################################
    # Model setting for U-net
    print('\033[96mConstruct U-net model. \033[0m')
    block_channels = [12,24,36,48,64] #  [64, 128, 256, 512, 1024] # original U-net
    block_conv = 2
    channels = 4 # how many band for channel (R,G,B,SWIR)
    loss = "DL" # using which loss function e.g. "DL" is Dice loss
    batch_size = 30
    epochs = 1
    # model_dest = "./experiments/台灣區決賽所用模型及預測結果/"
    model_dest = "./experiments/test/"
    args = ArgumentForLoss(model_dest, len(block_channels), block_conv, block_channels, channels,
             loss, batch_size, epochs)


    model = U_net(args.channels, args)
    model.compile(loss=loss_arg(args.loss)[1], optimizer='adam')
    print("\033[44mModel built.\033[0m\n")
    

    ################################################
    # Define the patch size and target tif file
    print("\033[96mLoad data\033[0m")
    target_tif_path = "./test.tif" # the target tif file
    training_path = "../data/Sentinel-2/training" # the folder of training data
    box_size = 128 # # the patch size of spilt image
    ref_path = [os.path.join(training_path, i) for i in os.listdir(training_path)]
    X, y, X_val, y_val = [], [], [], []
    
    start = time.time()
    for tif_name in ref_path:
        print("Process:", tif_name)
        tmp = load_data(tif_name, target_tif_path, args.box_size)
        X.append(tmp[0]); X_val.append(tmp[2])
        y.append(tmp[1]); y_val.append(tmp[3])
    X, y, X_val, y_val = map(np.vstack, [X, y, X_val, y_val])
    end = time.time()
    print('\033[44mProcessing total data done. Using {} sec.\033[0m'.format(end-start))
    print("All numbers of data:", X.shape)
    ################################################
    print("\033[96mStart to training\033[0m")
    print("Using Loss:", loss_arg(args.loss)[0])
    fit_training(X, y, X_val, y_val, model, loss_arg(args.loss)[1], args.batch_size, args.epochs, args)
    print('\033[44mEnd of training\033[0m')
    
    ################################################
    # evaluate each tif data by the training model:
    print("\033[96mStart to predict data\033[0m")
    using_data = "../data/Sentinel-2/projected_tif" # the folder which contains all the tif file we want to predict
    clip_shp_path = "../data/Sentinel-2/雲林縣/yunlin.shp" # clip the prediction result by the shp file 
    tif_list = [os.path.join(using_data, f) for f in os.listdir(using_data) if f.endswith('.tif')]
    evaluate(args.model_dest, tif_list, args, clip_shp_path=clip_shp_path)
    