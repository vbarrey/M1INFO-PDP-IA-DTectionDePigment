from pathlib import Path
import sys
from PIL import Image
import os
import json
import random
import config



BACKGROUD_MASK = 0
UNARY_MASK = 255

def isIn(x, y, w, h):
    return x>=0 and y>=0 and x<w and y<h

def spreadColor(img, ix, iy):
    w,h = img.size
    color = img.getpixel((ix,iy))
    marked =[]
    marked.append((ix,iy))
    while marked : 
        x, y = marked.pop(0)
        for dx in [-1,0,1] :
            for dy in [-1,0,1] :
                nx = x+dx
                ny=y+dy
                if isIn(nx, ny, w, h):
                    if img.getpixel((nx,ny)) == UNARY_MASK:
                        marked.append((nx,ny))
                        img.putpixel((nx,ny), color) 


def createMask(full_img, imageName, jsonPath):
    width, height = full_img.size
    
    
    jsonFile = open(jsonPath, 'r', encoding="utf-8")
    data=[]
    foundPos = False
    with jsonFile as file:
        data = json.load(file)
    n=0
    pixels = []
    for d in data:
        if "RGB" in d:
            for area in data[d]:
                pixels.append(json.JSONDecoder().decode(data[d][area]))

    mask = Image.new('L', full_img.size, BACKGROUD_MASK)

    for i in range(len(pixels)):
        for pix in pixels[i]:
            mask.putpixel(pix, UNARY_MASK)
            foundPos = True


    return mask, foundPos


    

def saveMasks(img, mask, sub_img_size, pp, bin, imageName, dest_dir) : 
    
    croppImages = croppedImages(img, sub_img_size)
    croppMasks = croppedImages(mask, sub_img_size)

    nbImg = len(croppImages)

    positive_index =[]
    for i in range(nbImg):
        if(sum(croppMasks[i].getextrema()) != 0) : #if not all black
            positive_index.append(i)

    negative_index = list(set(range(nbImg)).difference(set(positive_index)))  #set difference

    nbPos = len(positive_index)
    nbNeg = len(negative_index)

    targetNbNeg = int( (nbPos * (100-pp)) / pp )

    save_index =[]

    save_index.extend(positive_index)

    
    
   
    if (targetNbNeg < nbNeg) :
        if pp !=100 :
            save_index.extend(random.sample(negative_index, targetNbNeg))
    else :
        save_index.extend(negative_index)
    
    for i in save_index :
        croppImages[i].save(f"{dest_dir}/image/{imageName}_{i}.png")
        if i in positive_index or config.SAVE_EMPTY_MASKS:
            if i in positive_index and not bin:
                color_i = 1
                for xp in range(sub_img_size):
                    for yp in range(sub_img_size):
                        if  croppMasks[i].getpixel((xp,yp)) == UNARY_MASK :
                            croppMasks[i].putpixel((xp,yp), color_i)
                            spreadColor( croppMasks[i], xp, yp)
                            color_i += 1

            croppMasks[i].save(f"{dest_dir}/mask/M{imageName}_{i}.png")
        





def croppedImages(full_img, sub_img_size) :
    cropped_img = []
    x = 0
    
    width, height = full_img.size
    while x <width :
        y=0
        if x + sub_img_size >= width:
            x = width-sub_img_size
        while y <height :
            if y + sub_img_size >= height :
                y = height-sub_img_size
            cropped_img.append(full_img.crop((x, y, x + sub_img_size, y+sub_img_size)))
            y+=sub_img_size
        x+=sub_img_size
    return cropped_img


def generateDataset(sub_img_size, src, dest,pp,  bin):

    src_dir = src
    dest_dir = dest

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    if not os.path.exists(dest_dir+"/image/"):
        os.mkdir(dest_dir+"/image/")
    if not os.path.exists(dest_dir+"/mask/"):
        os.mkdir(dest_dir+"/mask/")
    src_files  = Path(src_dir+os.sep + "images").glob('*.jpg')
    for f in src_files :
        full_img = Image.open(f)
        imageName = Path(f).stem
        jsonPath = os.path.join(src_dir ,"jsons" , imageName + ".json")
        mask, foundPos = createMask(full_img,imageName, jsonPath)
        if foundPos:
            saveMasks(full_img, mask, sub_img_size, pp, bin, imageName, dest_dir)