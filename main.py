import os
import sys
import subprocess
import argparse
from pathlib import Path

def train(args) :
    sys.path.append("modules"+os.sep+"train"+os.sep+"include")
    sys.path.append("modules"+os.sep+"train")
    from modules.train.train_maskRCNN import trainMaskRcnn
    abs_datadir = args.dataset_dir.resolve()
    abs_odir = args.output_dir.resolve()
    os.chdir("."+os.sep+"modules"+os.sep+"train")
    trainMaskRcnn(str(abs_datadir), str(abs_odir), args.epochs, args.batch_size)

def runserver(args) :
    subprocess.call([sys.executable, os.path.join("modules", "web-labeler", "backend","manage.py"), 'runserver', '0.0.0.0:8000'])
    
def generateDataset(args):
    sys.path.append("modules"+os.sep+"dataset_generator")
    from modules.dataset_generator.dataset_generator import generateDataset
    abs_indir = args.input_dir.resolve()
    abs_odir = args.output_dir.resolve()
    os.chdir("."+os.sep+"modules"+os.sep+"dataset_generator")
    generateDataset(args.sub_image_size, str(abs_indir), str(abs_odir), args.positive_percentage, args.binary)

def eval(args): 
    from modules.eval.eval_maskRCNN import evalImages
    abs_indir = args.input_dir.resolve()
    abs_odir = args.output_dir.resolve()
    abs_mpath = args.model.resolve()
    os.chdir("."+os.sep+"modules"+os.sep+"eval")
    evalImages(str(abs_indir), str(abs_odir), str(abs_mpath), args.sub_image_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    

    # create sub-parser
    sub_parsers = parser.add_subparsers(help='sub-command help', required=True)

    # create the parser for the "runserver" sub-comman
    parser_runserver = sub_parsers.add_parser('runserver', help='sub command to start pixel labeler server')
    parser_runserver.set_defaults(func=runserver)

    # create the parser for the "generate-dataset" sub-command
    parser_dataset = sub_parsers.add_parser('generate-dataset', help='sub command to generate formatted dataset')
    parser_dataset.add_argument('-i', '--input-dir', type=Path, help='Path to the directory containing \'images\\\' and \'jsons\\\' subfolders', required=True)
    parser_dataset.add_argument('-o', '--output-dir', type=Path, help='Path to the directory where the new dataset will be written', required=True)
    parser_dataset.add_argument('-s', '--sub-image-size', type=int, help='Size of the images to be created for the dataset (Default 256)', required=False, default=256)
    parser_dataset.add_argument('-pp', '--positive-percentage', type=int, help='Percentage of positive images to include in the generated dataset (Default 100%%)', required=False, default=100)
    parser_dataset.add_argument('-b', '--binary', help='If set, mask will be binary (black/white), else each instance instance will be encoded with a different shade of grey',  default=False, action='store_true')
    parser_dataset.set_defaults(func=generateDataset)

    # create the parser for the "train" sub-command
    parser_train = sub_parsers.add_parser('train', help='sub command to train model')
    parser_train.add_argument('-d', '--dataset-dir', type=Path, help='Path to the dataset to use for training', required=True)
    parser_train.add_argument('-o', '--output-dir', type=Path, help='Path to the directory where the model and report will be written', required=True)
    parser_train.add_argument('-e', '--epochs', type=int, help='Number of epochs the model will be trained for', required=True)
    parser_train.add_argument('-b', '--batch-size', type=int, help='Number of epochs the model will be trained for', required=True)
    parser_train.set_defaults(func=train)

    # create the parser for the "eval" sub-command
    parser_eval = sub_parsers.add_parser('eval', help='sub command to evaluate images')
    parser_eval.add_argument('-i', '--input-dir', type=Path, help='Path to the directory containing images to evaluate', required=True)
    parser_eval.add_argument('-o', '--output-dir', type=Path, help='Path to the directory where the predicted masks will be written', required=True)
    parser_eval.add_argument('-m', '--model', type=Path, help='Pytorch model (.pt file) to use for evaluation', required=True)
    parser_eval.add_argument('-s', '--sub-image-size', type=int, help='Size of the cropped images to pass to the model (Default 256)', required=False, default=256)
    parser_eval.set_defaults(func=eval)

    args = parser.parse_args()
    args.func(args)



    