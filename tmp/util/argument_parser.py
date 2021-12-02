import argparse, os, ast, distutils.util

def arg_parser():
    parser = argparse.ArgumentParser()

    #---purpose---
    parser.add_argument("-train", default=False,type=distutils.util.strtobool,
                        help="Train network")
    parser.add_argument("-evaluate", default=False,type=distutils.util.strtobool,
                        help="Evaluate performance")

    #---training arguement
    parser.add_argument("-epochs", default='300', type=int,
                        help="Number of training epochs")
    parser.add_argument("-batch_size", default='30', type=int,
                        help="Number of training batch size")
    parser.add_argument("-loss", default='DL', type=str,
                        help="loss function: CE,DL,SS,WDL,WSS,WCE")
                        

    #---data parser---
    parser.add_argument("-using_data", default=None,
                        help="Directory name contained tif file want to predict")
    parser.add_argument("-box_size", default=128, type=int,
                        help="Patch size of tif spitted")
    parser.add_argument("-normalization", default=False, type=distutils.util.strtobool,
                        help="Using for data normalization")
    parser.add_argument("-model_dest", default='./experiments/台灣區決賽所用模型及預測結果/',
                        help="Directory path to model")
    parser.add_argument("-training_path", default='../data/Sentinel-2/training',
                        help="Path to training data")
    parser.add_argument("-target_path", default='./test.tif',
                        help="Path to target")
    parser.add_argument("-clip_shp_path", default=None,
                        help="Path to target")

    #---model structure---
    parser.add_argument("-blocks", default='5',type=int,
                        help="Number of down sampling layers")
    parser.add_argument("-block_conv", default='2',type=int,
                        help="Number of convolution layers in block")
    parser.add_argument("-block_channels", default='[12,24,36,48,64]',type=str,
                        help="Number of convolution layers in block")
    parser.add_argument("-dropout_rate", default='0',type=float,
                        help="Drop out rate between each down sampling blocks")
    parser.add_argument("-channels", default='4', type=int,
                        help="Number of input channels/bands")


    args = parser.parse_args()

    if args.evaluate == True and args.pred_dest == None:
        args.pred_dest = args.model_dest + "/prediction"

    args.block_channels = ast.literal_eval(args.block_channels)
    args.blocks = len(args.block_channels)

    # create using folder
    if not os.path.exists(args.model_dest):
        os.makedirs(args.model_dest)
    
    # if args.pred_dest != None and not os.path.exists(args.pred_dest + '/test/grid/input'):
    #     for i in ["train","test","validation"]:
    #         for j in ["input","prediction","target"]:
    #             os.makedirs("{}/{}/grid/{}".format(args.pred_dest,i,j))

    return args
