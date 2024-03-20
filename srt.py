import os
import torch
import numpy
import torchvision.transforms
from mit_semseg.models import ModelBuilder, SegmentationModule
import asyncio
import time
from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
from mit_semseg.utils import colorEncode
import cv2
import numpy as np
from torch import nn
import tensorflow as tf
import tensorflow_hub as hub


# ************************************************GLOBAL DECLARATIONS START************************************************
image_size = 480
# Download the model from TF Hub.
model = hub.load('https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-thunder/versions/4')
movenet = model.signatures['serving_default']

EDGES = {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (0, 5): 'm',
            (0, 6): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
        }


# ************************************************GLOBAL DECLARATIONS END************************************************

def get_keypoints(image):

    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, 256, 256)
    input_image = tf.cast(input_image, tf.int32)

    # Run model inference.
    keypoints_with_scores = movenet(input_image)['output_0'].numpy()

    # print(keypoints_with_scores)

    # Visualize the predictions with image.
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(
        display_image, image.shape[0], image.shape[1]), dtype=tf.int32)


    original_array_p1 = np.squeeze(keypoints_with_scores)[:, 0:1] * image.shape[0]
    original_array_p2 = np.squeeze(keypoints_with_scores)[:, 1:2] * image.shape[1]


    original_array_p1 = original_array_p1.astype(int)
    original_array_p2 = original_array_p2.astype(int)


    original_array = np.concatenate((original_array_p1, original_array_p2), axis=1)

    original_array[:, [0, 1]] = original_array[:, [1, 0]]
    original_array = original_array.astype(int)

    indices_to_extract = [9, 10, 13, 14]
    original_array = original_array[indices_to_extract]

    return original_array, keypoints_with_scores, display_image



def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 2)


    return frame


def grayscale_to_rgb(image_gray):
    """
    Convert a grayscale 2D image to a 3D RGB image.
    
    Parameters:
        image_gray (ndarray): Input grayscale image (2D array).
        
    Returns:
        ndarray: Output RGB image (3D array).
    """
    # Ensure input image has two dimensions
    if image_gray.ndim != 2:
        raise ValueError("Input image must be a 2D array (grayscale)")
    
    # Convert grayscale to RGB by duplicating intensity values across three channels
    image_rgb = np.repeat(image_gray[:, :, np.newaxis], 3, axis=2)
    
    return image_rgb


# ( 345, 12)
def check_above_below(img, pt):


    black_flag = 1
    white_flag = 1

    for i in range(200):
        try:
            if img[pt[1], pt[0] - i] != 0:
                black_flag = 0

        except:
            pass

        try:
            if img[pt[1], pt[0] + i] != 254:
                white_flag = 0
        except:
            pass

    # 2 for accepted point
    # < 2 for rejected point
    if black_flag == 1 and white_flag == 1:
        return True
    
    else:
        return False
    


def srt_get_metric(video_path):

    # video_path = 'sample1.mp4'
    # try:

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    # segmentation_module.cuda()


    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])

    # video_path = "./rydvvid6.mp4"

    # video_path = sys.argv[1]
    output_video_path = video_path[:-4] + '_output' + '.mp4'


    def read_video_and_array(video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Initialize an empty list to store frames
        frames_list = []

        # Read frames from the video until it's finished
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                # Convert frame to numpy array and append it to the list
                frames_list.append(frame)
            else:
                break

        # Release the video capture object
        cap.release()

        # Convert the list of frames to a numpy array
        frames_array = np.array(frames_list)

        return frames_array


    frames = read_video_and_array(video_path)


    # pil_image = PIL.Image.open('floor.jpg').convert('RGB')
    # img = cv2.imread('pred.jpg')
    img_original = numpy.array(frames[0])

    # img_original = numpy.array(img)

    # print(img_original.shape)
    img_data = pil_to_tensor(img_original)
    singleton_batch = {'img_data': img_data[None]}
    output_size = img_data.shape[1:]


    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)
        
    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()

    # print(f'final pred {pred}')


    cv2.imwrite("region.jpeg", pred[0: 100, 0: 300])
    

    floor_coords = []
    min_dim1 = pred.shape[0]
    min_dim2 = pred.shape[1]
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
        
            # 3 is the pixel label that belongs to "floor"
            
            if pred[i, j] == 3:

                # finding the minimum dimensions of the x and y coords
                if j < min_dim2:
                    min_dim2 = j

                if i < min_dim1:
                    min_dim1  = i

                # checking the pixels that have the value 3, and making them completely white (254, 0, 0)
                pred[i, j]= 254
                # appending all the coords that belong to "floor" (have the value 3)
                floor_coords.append([i, j])

            else:
                pred[i, j] = 0

    floor_coords = np.array(floor_coords)

    # get threshold point
    # height, width


    # geting the max heiht value (lowest in the frame)
    min_y = np.max(floor_coords[:, 0])
    print(f'Initial min_y is {min_y}')
    print(f'floor coords {floor_coords}')



    saved_coords = []
    for coord in floor_coords:
        if coord[0] < min_y:
            # if np.all(pred[coord[1], coord[0] - 200 : coord[0]] == 0) and np.all(pred[coord[1], coord[0]: coord[0] + 200] == 254):
            #     min_y = coord[0]
            #     saved_coords = coord

            if check_above_below(pred, coord):
                min_y = coord[0]
                saved_coords = coord

    


    print(f'Final min_y is ')
    print(min_y)
            
    cv2.imwrite('./pred.jpg', pred)
    print(f'Segmented floor image written!')
    
    cap = cv2.VideoCapture(video_path)
    pred_rgb = grayscale_to_rgb(pred)

    # print(f'Showing pred_RGB shape')
    # print(pred_rgb.shape)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (pred_rgb.shape[1],pred_rgb.shape[0]))
    # t = 0
    metric_points = 10
    joints_touched = {}

    while True:
        ret, frame = cap.read()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 255)  # White color in BGR
        font_thickness = 2
        

        # t += 1

        if not ret:
            print("Error: Could not read frame.")
            break

        original_array, keypoints_with_scores, display_image = get_keypoints(frame)

        frame *= 0

        # draw_connections(frame, keypoints_with_scores, EDGES, 0.4)

        frame += np.uint8(pred_rgb)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 255)  # red color in BGR
        font_thickness = 2



        cv2.putText(frame, str(metric_points), (50, 50), font, font_scale, font_color, font_thickness)
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        # 443
        for i in range (original_array.shape[0]):

            # coords of keypoints (x, y)
            point_list = original_array[i].tolist()
            cv2.circle(frame, original_array[i], 5, (0, 0, 255), -1)

            # eg point_list -> (56, 176)
            # if y axis coord of point_list (contact point) goes below threshold y coord
            if point_list[1] >= min_y:
                if i not in joints_touched.keys():
                    joints_touched[i] = True
                    metric_points -= 1
                    cv2.circle(frame, original_array[i], 15, (0, 0, 255), 1)
                    # cv2.imwrite(f"{i}_contact.jpeg", frame)
                cv2.circle(frame, original_array[i], 15, (0, 0, 255), 1)

        # print(f'frame {t}')
        # cv2.imshow('frame', frame)
        out.write(frame)

    out.release()
    cap.release()



    print(f'Points: {str(metric_points)}')
    return str(metric_points)



srt_get_metric("") # video path




