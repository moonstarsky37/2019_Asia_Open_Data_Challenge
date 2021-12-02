import time, os, sys
from util.ziyu_model import *
from util.train import *
from util.data_parser import *
from util.argument_parser import *
from util.evaluate import *

if __name__ == '__main__':
    args = arg_parser()
    if args.train:
        ################################################
        # Model setting for U-net
        print('\033[96mConstruct U-net model. \033[0m')
        # e.g.
        # block_channels = [12,24,36,48,64]
        # block_channels = [64, 128, 256, 512, 1024] # original U-net
        # channels = 4 # how many band for channel (R,G,B,SWIR)
        # loss = "DL" # using which loss function e.g. "DL" is Dice loss
        model = U_net(args.channels, args)
        model.compile(loss=loss_arg(args.loss)[1], optimizer='adam')
        print("\033[44mModel built.\033[0m\n")
        

        ################################################
        # Define the patch size and target tif file
        print("\033[96mLoad data\033[0m")
        target_tif_path = args.target_path #"./test.tif"
        box_size = args.box_size # 128
        # Define the original tif file
        ## case 1: if all band in one tif file
        # ref_path = ["../data/Sentinel-2/projected_tif/T50QRM_20180322T022651_B2348.tif"]
        ref_path = [os.path.join(args.training_path, i) for i in os.listdir(args.training_path)]
        X, y, X_val, y_val = [], [], [], []
        
        start = time.time()
        for tif_name in ref_path:
            print("Process:", tif_name)
            tmp = load_data(tif_name, target_tif_path, box_size, normalizztion=args.normalization)
            X.append(tmp[0]); X_val.append(tmp[2])
            y.append(tmp[1]); y_val.append(tmp[3])
        X, y, X_val, y_val = map(np.vstack, [X, y, X_val, y_val])
        end = time.time()
        print('\033[44mProcessing total data done. Using {} sec.\033[0m'.format(end-start))
        ## case 2: if band in multiple tif file but in one folder
        # ref_path = "../data/Sentinel-2/T50QRM_20180322_FullBand"
        # X, y, X_val, y_val = load_data_from_folder(ref_path, target_tif_path, box_size)
        ## case 3: multiple folder
        # ref_path = ["../data/Sentinel-2/T50QRM_20180322_FullBand", "../data/Sentinel-2/S2A_MSIL1C_20180312_FullBand"]
        # X, y, X_val, y_val = load_data_from_folder_list(ref_path, target_tif_path, box_size)
        print("All numbers of data:",X.shape)
        ################################################
        print("\033[96mStart to training\033[0m")
        print("Using Loss:", loss_arg(args.loss)[0])
        fit_training(X, y, X_val, y_val, model, loss_arg(args.loss)[1], args.batch_size, args.epochs, args)

    elif args.evaluate:
        tif_list = [os.path.join(args.using_data, file) for file in os.listdir(args.using_data) if file.endswith('.tif')]
        evaluate(args.model_dest, tif_list, args)
    else:
        pass
