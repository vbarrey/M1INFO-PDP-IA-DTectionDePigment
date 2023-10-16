import os
from PIL import Image
import torch
from torchvision.transforms import functional as F
from pathlib import Path
import numpy as np
from PIL import ImageFilter

def toTensor(img):
    image = F.pil_to_tensor(img)
    image = F.convert_image_dtype(image)
    image.div(255)
    return image

def getSmallPredMask(img, model) :
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        model.to(device)
        model.eval()
        predMask = model([toTensor(img).to(device)])

    
    nb_masks = predMask[0]['masks'].size()[0]
    if(nb_masks == 0) :
        return Image.new('L', img.size, 0)
    allPredMasks = np.zeros(np.shape(predMask[0]['masks'][0, 0]))

    for i in range(nb_masks) :
        maskArr = predMask[0]['masks'][i, 0].cpu().numpy()
        allPredMasks = np.maximum(allPredMasks, maskArr)
    maskarray = (allPredMasks*255).astype(int)
    maskarray[maskarray>255] = 255
    i = Image.fromarray(maskarray)
    return i




    
def getFullPredMask(full_img, model, sub_img_size) :
    mask = Image.new('L', full_img.size, 0)

    
    x = 0
    
    width, height = full_img.size
    while x <width :
        y=0
        if x + sub_img_size >= width:
            x = width-sub_img_size +1
        while y <height :
            if y + sub_img_size >= height :
                y = height-sub_img_size +2 
            cropped_img = full_img.crop((x, y, x + sub_img_size, y+sub_img_size))
            cropped_mask = getSmallPredMask(cropped_img, model).convert('L')

            mask.paste(cropped_mask, (x,y))
            y+=sub_img_size-2
        x+=sub_img_size-1
    #mask = mask.filter(ImageFilter.MinFilter(3)) #erosion
    #mask = mask.filter(ImageFilter.MaxFilter(3)) #dilatation
    return mask



def evalImages(inputdir, outputdir, modelpath, size) :
    src_dir = inputdir
    dest_dir = outputdir

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    src_files  = Path(src_dir).glob('*.jpg')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(modelpath, map_location=device)

    model.eval()

    for f in src_files :
        full_img = Image.open(f)
        imageName = Path(f).stem
        full_mask = getFullPredMask(full_img, model, size)
        full_mask.save(dest_dir + os.sep + "PM" + imageName + ".png")