import os
import sys
sys.path.append("./include/")
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch
import utils
import transforms as T
from torch.optim.lr_scheduler import StepLR
import config
from html_report import HTMLReport
from datetime import datetime
import time
import matplotlib.pyplot as plt


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, size, transforms=None):
        self.root = root
        self.img_size = size
        self.transforms = transforms
        self.original_masks = []
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = []
        self.boxes_array = []
        for f in list(sorted(os.listdir(os.path.join(root, "image")))) :
            boxes = []
            img_path = os.path.join(self.root, "image", f)
            maskpath = img_path.replace("image" + os.sep, "mask"+os.sep+"M")
            mask = Image.open(maskpath)
            mask = np.array(mask)
            # instances are encoded as different colors
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set
            # of binary masks
            masks = []
            for id in obj_ids :
                masks.append([p == id for p in mask])
            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if xmin != xmax and ymin != ymax :
                    boxes.append([xmin, ymin, xmax, ymax])
            if(len(boxes)!=0):
                self.boxes_array.append(boxes)
                self.imgs.append(f)
                self.original_masks.append(maskpath)

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "image", self.imgs[idx])
        #mask_path = os.path.join(self.root, "mask", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        target = {}
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        maskpath = img_path.replace("image" + os.sep, "mask"+os.sep+"M")
        try :
            mask = Image.open(maskpath)

        except :
            mask = append(Image.new('L', (self.img_size,self.img_size), 0))

        #target["original_mask"] = mask
        self.original_masks[idx] = maskpath
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = []
        for id in obj_ids :
            masks.append([p == id for p in mask])
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        boxes = self.boxes_array[idx]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
       
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def getImageSize(dataset_dir) :
    img_name = os.listdir(os.path.join(dataset_dir, "image"))[0]
    img_path = os.path.join(dataset_dir, "image", img_name)
    img = Image.open(img_path).convert("RGB")
    return img.size[0]



def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform():

    return T.Compose([T.PILToTensor()])

###
# From https://stackoverflow.com/questions/306316/determine-if-two-rectangles-overlap-each-other
# By Pedro Gimeno
###
def bboxOverlap(bbox1, bbox2) : 
    return max(bbox1[0], bbox2[0]) < min(bbox1[2], bbox2[2]) and max(bbox1[1], bbox2[1]) < min(bbox1[3], bbox2[3])

def getPredictionResults(test_dataset, model, device) :
    total_fp = 0
    total_fn = 0
    total_tp = 0
    
    for origImage, origMask in test_dataset :

        with torch.no_grad():
            predMask = model([origImage.to(device)])
 
        nb_target = len(origMask["boxes"])
        nb_pred = len(predMask[0]['boxes'])

        fn = np.full((nb_target), True, dtype=bool)
        fp  = np.full((nb_pred), True, dtype=bool)

        for t_box_i in range(nb_target) : 
            
            for p_box_i in range(nb_pred) :

                if bboxOverlap(origMask["boxes"][t_box_i], predMask[0]['boxes'][p_box_i]) :
                    fn[t_box_i] = False
                    fp[p_box_i] = False

        nb_fp = np.count_nonzero(fp)
        nb_fn = np.count_nonzero(fn)
        nb_tp = nb_target - nb_fn

        #remove duplicate fp

        for p_box_i in range(nb_pred) :

            for p_box_j in range(p_box_i+1, nb_pred) :
                if fp[p_box_i] and fp[p_box_j] and bboxOverlap(predMask[0]['boxes'][p_box_i],predMask[0]['boxes'][p_box_j]) :
                    nb_fp -= 1
        total_fn += nb_fn
        total_fp += nb_fp
        total_tp += nb_tp
    return total_fn, total_fp, total_tp

def getExample(test_dataset, model, original_masks_paths,  device, idx):
    origImage, origMask = test_dataset[idx]
    #print(origMask)
    #for i in origMask['masks'].numpy()[0] :
        #print(i)
    with torch.no_grad():
        predMask = model([origImage.to(device)])
    # initialize our figure
    #print(origMask['masks'])
    nb_masks = predMask[0]['masks'].size()[0]
    threshold = 0.5
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    if(nb_masks == 0):
        print("ZEROOOOOOOOOOOO")
    allPredMasks = np.zeros(np.shape(predMask[0]['masks'][0, 0]))
    
    for i in range(nb_masks) :
        maskArr = predMask[0]['masks'][i, 0].cpu().numpy()
        maskArr[maskArr < threshold] = 0.0
        allPredMasks = np.maximum(allPredMasks, maskArr)
        
    nb_target = len(origMask["boxes"])
    nb_pred = len(predMask[0]['boxes'])

    fn = np.full((nb_target), True, dtype=bool)
    fp  = np.full((nb_pred), True, dtype=bool)

    for t_box_i in range(nb_target) : 
        
        for p_box_i in range(nb_pred) :

            if bboxOverlap(origMask["boxes"][t_box_i], predMask[0]['boxes'][p_box_i]) :
                fn[t_box_i] = False
                fp[p_box_i] = False

    nb_fp = np.count_nonzero(fp)
    nb_fn = np.count_nonzero(fn)
    nb_tp = nb_target - nb_fn

    #remove duplicate fp

    for p_box_i in range(nb_pred) :

        for p_box_j in range(p_box_i+1, nb_pred) :
            if fp[p_box_i] and fp[p_box_j] and bboxOverlap(predMask[0]['boxes'][p_box_i],predMask[0]['boxes'][p_box_j]) :
                nb_fp -= 1
    ax[0].imshow(Image.fromarray(origImage.mul(255).permute(1, 2, 0).byte().numpy()))
    om =np.array(Image.open(original_masks_paths[test_dataset.indices[idx]]))
    om[om != 0] = 255
    ax[1].imshow(Image.fromarray(om.astype(float)))
    ax[2].imshow(Image.fromarray(allPredMasks*255))
    ax[0].set_title("Image")
    ax[1].set_title(f"Original Mask - expected : {nb_target}")
    ax[2].set_title(f"Predicted Mask - fn:{nb_fn} - fp:{nb_fp} - tp:{nb_tp}")
    # set the titles of the subplots
    
    # set the layout of the figure and display it
    figure.tight_layout()
    return figure

def confusionMatrix(fn, fp, tp) :
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    conf_matrix = np.array([[0,fp],[fn, tp]])
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(2):
        for j in range(2):
            if i!=0 or j != 0 :
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    ax.set_xlabel('Predicted', fontsize=18)
    ax.set_ylabel('Expected', fontsize=18)
    ax.set_title('Confusion Matrix', fontsize=18)
    return fig


def getPrecision(fn, fp, tp) :
    return tp / float(fp+tp)

def getRecall(fn, fp, tp) :
    return tp / float(fn+tp)

def getLossGraph(loss_for_epoch):
    e = np.arange(1, len(loss_for_epoch)+1)
    

    
    fig, ax = plt.subplots()
    ax.plot(e, loss_for_epoch, '-ok')

    ax.set(xlabel='epoch ', ylabel='loss',
        title='Evolution of loss')
    ax.grid()
    return fig



def trainMaskRcnn(dataset_dir, ouputdir, epochs, batch_size) :
    img_size = getImageSize(dataset_dir)
    dataset = PennFudanDataset(dataset_dir, img_size, get_transform())
    dataset_test = PennFudanDataset(dataset_dir, img_size, get_transform())
    original_masks_test = dataset.original_masks
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_size = len(dataset)
    train_size = int(dataset_size*config.RATIO_TRAIN)
    dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[train_size:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:500"
    loss_for_epoch = []
    start_time = time.time()
    best_loss = 1000
    if not os.path.exists(ouputdir) :
        os.mkdir(ouputdir)
    for epoch in range(epochs):
        # train for one epoch, printing every 10 iterations
        loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100000)
        loss_for_epoch.append(loss)
        if loss < best_loss :
            best_loss = loss
            print("Saving Model")
            torch.save(model, ouputdir+os.sep+'model.pt')
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
    
    training_time_min = (time.time() - start_time)/60
    

    

    model.eval()

    print("Creating Report")
    print(len(dataset_test))
    fn, fp, tp = getPredictionResults(dataset_test, model, device)
    
    example_figs = []
    
    for i in range(0,10) :
        example_figs.append(getExample(dataset_test, model, original_masks_test, device,  i))
    
    report = HTMLReport(f"{datetime.now():%Y-%m-%d %H:%M:%S%z}",
     dataset_dir, 
     epochs, 
     batch_size, 
     img_size, 
     train_size, 
     dataset_size-train_size,
     training_time_min, 
     getLossGraph(loss_for_epoch), 
     confusionMatrix(fn, fp, tp), 
     getPrecision(fn, fp, tp), 
     getRecall(fn, fp, tp), 
     example_figs)
    
    report.save(ouputdir)
    print("Done.")



