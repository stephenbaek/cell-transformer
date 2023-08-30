# Import packages:
import os, sys, random, struct, shutil, mmcv, scipy, mmcv, skimage, imageio

import cv2 as cv
import numpy as np
from scipy import ndimage

from skimage import img_as_ubyte
import skimage.io as io

from PIL import Image

from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
plt.rcParams["figure.figsize"] = (10,10)
plt.rcParams.update({'font.size':10})

config_file = os.getcwd() + '/configs/swin-sim-seg/1_mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sim_seg.py'
checkpoint_file = os.getcwd() + '/saved_model/epoch_50.pth'
model = init_detector(config_file,checkpoint_file)

# Obtain the cell probability threshold from provided parameters
cell_prob_param = float(sys.argv[1])
print(cell_prob_param)

# Scan the current images, and generate the corresponding folder in predictions:
image_list = []
save_directory = os.getcwd() + '/predictions/'
for root, dirs, files in os.walk(os.getcwd() + '/data/'):
    for file in files:
        if file.endswith(".tif") and "ZSeries" in file and "gt" not in file:
            print(file)
            image_list.append(os.path.join(root, file))
            folder_name = root.split('data')[1].split('/')[1]
            if not os.path.exists(os.path.join(save_directory, folder_name)):
                os.makedirs(os.path.join(save_directory, folder_name)) 
image_list.sort()
print(len(image_list))   

# Post-prcessing: Remove small or large predictions, and predictions near the edges
def refine_the_predicitions(bbox_collection, seg_collection):
    refined_seg_collection = []
    refined_bbox_collection = []
    refined_seg_idx = 0
    
    for current_box in range(0, bbox_collection.shape[0]):
        if bbox_collection[current_box, 2] < 510 and bbox_collection[current_box, 3] < 510 and bbox_collection[current_box, 0] > 2 and bbox_collection[current_box, 1] > 2 and bbox_collection[current_box, 4] > cell_prob_param  and (bbox_collection[current_box, 3] - bbox_collection[current_box, 1])<100 and (bbox_collection[current_box, 3] - bbox_collection[current_box, 1]) > 10 and (bbox_collection[current_box, 2] - bbox_collection[current_box, 0])<100 and (bbox_collection[current_box, 2] - bbox_collection[current_box, 0])>10:
            #print(bbox_collection[current_box, 2], bbox_collection[current_box, 3])
            refined_seg_collection.append(seg_collection[current_box,:,:])
            refined_bbox_collection.append(bbox_collection[current_box,:])
    refined_seg_collection = np.asarray(refined_seg_collection)
    refined_bbox_collection = np.asarray(refined_bbox_collection)
    return refined_bbox_collection, refined_seg_collection

# Post-prcessing: Remove some predictions which overlap with each other
def post_processing_the_predictions(refined_seg_collection):
        post_proc_seg_collection = []
        dup_cell = []
        
        for index1 in range(0, refined_seg_collection.shape[0]):
            if index1 not in dup_cell:
                first_cell = refined_seg_collection[index1,:,:]>0
                first_cell = np.atleast_1d(first_cell)
                max_iou = 0
                for index2 in range(index1+1, refined_seg_collection.shape[0]):
                    if index2 not in dup_cell:
                        second_cell = refined_seg_collection[index2,:,:]>0
                        second_cell = np.atleast_1d(second_cell)

                        intersection = np.count_nonzero(first_cell & second_cell)
                        size_i1 = np.count_nonzero(first_cell)
                        size_i2 = np.count_nonzero(second_cell)

                        try:
                            iou = intersection / float(size_i1 + size_i2 - intersection)
                            if iou > max_iou:
                                max_iou = iou

                        except ZeroDivisionError:
                            iou = 1.0
                        if iou>0.2: #20% overlap allowed between predictions
                            dup_cell.append(index2)
                post_proc_seg_collection.append(refined_seg_collection[index1,:,:])
        post_proc_seg_collection = np.asarray(post_proc_seg_collection)
        return post_proc_seg_collection

# Save the predictions in a figure:      
def save_segmentation_results():
    for number in range(0, len(image_list)): 
        current_pred_image_cell_collection = []
        
        filename = image_list[number]
        img = filename
        
        bbox_collection = inference_detector(model,img)[0][0]
        bbox_collection = np.asarray(bbox_collection)
        seg_collection = inference_detector(model,img)[1][0]
        seg_collection = np.asarray(seg_collection)
        
        
        """ Refine the predictions:"""
        refined_bbox_collection, refined_seg_collection = refine_the_predicitions(bbox_collection, seg_collection)
        
        """ Post-processing the predictions:"""
        post_proc_seg_collection = post_processing_the_predictions(refined_seg_collection)

        """ Generate the figures from post-processing: """
        if post_proc_seg_collection.shape[0]!=0:
            
            """Save the numpy array"""
            output_image_name = os.path.join(save_directory, filename.split('/')[-3], filename.split('/')[-1])
            output_numpy_name = output_image_name[:-4] + '.npy'
            np.save(output_numpy_name, post_proc_seg_collection)
        else:
            print('---------------------------------------------------------------')
            print('delete %s'%filename)
save_segmentation_results()            

# Save the ROI Zip files to be read by ImageJ/Fiji
def roiexport(filename, x, y, slice_pos, name):
    """Exports an ImageJ roi file.

    Arguments:
        filename: file path to the roi file
        x: list of x coordinates
        y: list of y coordinates (len(y) == len(x))
        slice_pos: z slice number
        name: name of the roi

    Returns:
        N/A
    """
    with open(filename, 'wb') as file:
        file.write(struct.pack('>ssss', b'I',b'o',b'u',b't'))   # 0-3  Iout
        file.write(struct.pack('>h', 227))                      # 4-5 version
        file.write(bytes([7])) # TYPE = 7  polygon=0; rect=1; oval=2; line=3; freeline=4; polyline=5; noRoi=6; freehand=7; traced=8; angle=9, point=10;
        file.write(bytes([0]))
        top = min(y)
        left = min(x)
        bottom = max(y)
        right = max(x)
        ncoords = len(x)
        file.write(struct.pack('>h', top))   # 8-9
        file.write(struct.pack('>h', left))  # 10-11
        file.write(struct.pack('>h', bottom)) # 12-13
        file.write(struct.pack('>h', right)) # 14-15
        file.write(struct.pack('>h', ncoords)) # 16-17
        file.write(struct.pack('>ffff', 0.0, 0.0, 0.0, 0.0))  # 18-33  X1, Y1, X2, Y2
        file.write(struct.pack('>h', 0))  # 34-35 Stroke width
        file.write(struct.pack('>f', 0.0))  # 36-39 ROI_size
        for i in range(4):
            file.write(bytes([0])) # 40-43 Stroke Color = 0 0 0 0
        for i in range(4):
            file.write(bytes([0])) # 44-47 Fill Color = 0 0 0 0
        file.write(struct.pack('>h', 0)) # 48-49 subtype = 0
        file.write(struct.pack('>h', 0)) # 50-51 options = 0
        file.write(bytes([0])) # 52 arrow stype or aspect ratio = 0
        file.write(bytes([0])) # 53 arrow head size = 0
        file.write(struct.pack('>h', 0)) # 54-55 rounded rect arc size = 0, 0
        file.write(struct.pack('>i', slice_pos))   # 56-59 position = (0, 20)
        h2offset = 4*len(x)+64
        file.write(struct.pack('>i', h2offset))    # 60-63 header2 offset
        for xcoord in x:
            file.write(struct.pack('>h', xcoord - left))
        for ycoord in y:
            file.write(struct.pack('>h', ycoord - top))

        # Header 2
        file.write(bytes([0,0,0,0]))  # 0-3
        file.write(struct.pack('>iii', 0, 0, 0))  # 4-7 C_POSITION, 8-11 Z_POSITION, 12-15 T_POSITION
        file.write(struct.pack('>i', h2offset + 64))   # 16-19 name offset
        file.write(struct.pack('>i', len(name)))   # 20-23 name length
        file.write(struct.pack('>i', 0))   # 24-27 OVERLAY_LABEL_COLOR
        file.write(struct.pack('>h', 0))   # 28-29 OVERLAY_FONT_SIZE

        file.write(bytes([0]))   # 30 GROUP 
        file.write(bytes([0]))   # 31 IMAGE_OPACITY 
        file.write(struct.pack('>i', 0))   # 32-35 IMAGE_SIZE 
        file.write(struct.pack('>f', 0.0))   # 36-39 FLOAT_STROKE_WIDTH 
        file.write(struct.pack('>f', 0.0))   # 40-43 ROI_PROPS_OFFSET  
        file.write(struct.pack('>f', 0.0))   # 44-47 ROI_PROPS_LENGTH   
        file.write(struct.pack('>f', 0.0))   # 48-51 COUNTERS_OFFSET
        for i in range(52, 64):
            file.write(bytes([0]))  # 52-63
        for i in range(len(name)):
            file.write(struct.pack('>h', ord(name[i])))
def shoelace(coords):
    corners = np.array([coords[0], coords[1]]).transpose()
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def find_contours(amask):
    temp = amask.astype('uint8')
    contours, hierarchy = cv.findContours(temp, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[-2:]
    return contours

def find_coordinates(aContour):
    #print(aContour)
    temp = aContour.reshape(aContour.shape[0],2)
    x_coord = temp[:,0]
    y_coord = temp[:,1]
    return x_coord, y_coord
def centroid(arr1, arr2):
    length = arr1.shape[0]
    w = np.max(arr1) - np.min(arr1)
    h = np.max(arr2) - np.min(arr2)
    sum_x = np.sum(arr1)
    sum_y = np.sum(arr2)
    return sum_x/length, sum_y/length, w, h
  
def generate_zip(current_cell, current_cell_index, roi_path, result_path):
    contours = find_contours(current_cell)
    if (len(contours)>1):
        area_lst = []
        for acontour in contours:
            area_lst.append(shoelace(find_coordinates(acontour)))

        max_idx = area_lst.index(max(area_lst))
        valid_contour = contours[max_idx]
        valid_area = area_lst[max_idx]
    elif len(contours)!=0: 
        valid_contour = contours[0]
        valid_area = shoelace(find_coordinates(valid_contour))

    x_coords, y_coords = find_coordinates(valid_contour)
    z_coords = 0#stack_num
    cen_info = centroid(x_coords, y_coords)
    roi_name = '%03d-%03d-%03d'%(cen_info[0], cen_info[1], z_coords)
    current_cell_index = '%03d'%current_cell_index
    roiexport(roi_path + '/' + str(current_cell_index) +'.roi', list(x_coords), list(y_coords), z_coords, roi_name)
    shutil.make_archive(roi_path, 'zip', roi_path)
    
    
print('start generating roi')
for current_name in image_list:
    current_masks_path = save_directory + current_name.split('data')[1].split('Z-Projection')[0] + current_name.split('Z-Projection')[1][:-4] + '.npy'
    if not os.path.exists(current_masks_path):
        continue
    #print(current_masks_path)
    current_masks = np.load(current_masks_path)
    print(current_masks.shape)
    
    stack_num = current_name.split('Stack_')[1].split('_')[0]
    stack_num = int(stack_num) // 10
    #print(stack_num)
    result_dir_prefix = save_directory + '/' + current_name.split('data')[1].split('Z-Projection')[0]
    if not os.path.exists(result_dir_prefix):
        os.makedirs(result_dir_prefix)
    
    roi_path = result_dir_prefix + '/RoiSet' + '_' + str(stack_num)
    if not os.path.exists(roi_path):
        os.makedirs(roi_path)
    
    for current_cell_index in range(0, current_masks.shape[0]):
        current_cell = current_masks[current_cell_index,:,:]
        #print(current_cell_index)
        generate_zip(current_cell, current_cell_index, roi_path, result_dir_prefix)
