from ultralytics import YOLO
import torch
import cv2
import sys
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
# from segment_anything import sam_model_registry, SamPredictor



# # function to show SAM model
# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)
    
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

# from segment_anything import sam_model_registry, SamPredictor

# sam_checkpoint = "F:\Leaf_Detection\SAM\check_point\sam_vit_b_01ec64.pth"
# model_type = "vit_b"

# device = "cuda"

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

# predictor = SamPredictor(sam)




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detect",
    default="F:\Leaf_Detection\weight\mAP_0.587_50epoch_4batch_yolov8x-p2_dataset_part1\\best.pt",
	help="path to model weight matrix")
ap.add_argument("-s", "--segment",
    default="F:\Leaf_Detection\weight\segment_model_mAP_0.961_100epoch_8batch_yolov8m_dataset_for_segment_v2\\best.pt",
	help="path to model weight matrix")
ap.add_argument("-i", "--input", required=True,
	help="path to input source")
args = vars(ap.parse_args())

# load the model from path
detect_model = YOLO(args['detect'])   
                            # F:\\Leaf_Detection\\weight\\map0.584_100epoch_8batch_yolov8x_above_view_add_more_data\\best.pt
                            # sys.argv[1]
# print(torch.cuda.is_available())

def detect_image_id(image):
    '''Input image and return the pixel scale of the image and image id'''
    
    
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    corner, id, rejected = detector.detectMarkers(image)
    if len(corner) != 1:
        sys.exit('Marker number not valid. Please! try again')
    corner = corner[0]
    id = int(id)
    corner = corner.reshape((4, 2))
    # (topLeft, topRight, bottomRight, bottomLeft) = corner
    # convert each of the (x, y)-coordinate pairs to integers
    # topRight = (int(topRight[0]), int(topRight[1]))
    # bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    # bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    # topLeft = (int(topLeft[0]), int(topLeft[1]))
    #length edge of marker
    # edge_1 = math.sqrt(pow(topRight[0] - topLeft[0], 2) + pow(topRight[1] - topLeft[1], 2))
    # edge_2 = math.sqrt(pow(topLeft[0] - bottomLeft[0], 2) + pow(topLeft[1] - bottomLeft[1], 2))
    # marker_area = edge_1 * edge_2
    # print('marker_area: ',marker_area, ' pixel')
    
    # shoelace formula
    marker_area = 0.5 * abs(sum((corner[i][0] * corner[i + 1][1]) - (corner[i + 1][0] * corner[i][1])
                    for i in range(len(corner) - 1))
                    + (corner[-1][0] * corner[0][1]) - (corner[0][0] * corner[-1][1]))
    print('marker_area: ',marker_area, ' pixel')
    pixel_scale = 9/marker_area
    
    print('Pixel scale: ' , pixel_scale, ' cm^2/pixel')
    print('Image ID: ', id)
    
    
    (topLeft, topRight, bottomRight, bottomLeft) = corner
    # convert each of the (x, y)-coordinate pairs to integers
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))
    
    img = image.copy()
    # draw the bounding box of the ArUCo detection
    cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
    cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
    cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
    cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)
    # compute and draw the center (x, y)-coordinates of the ArUco
    # marker
    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
    cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)
    # draw the ArUco marker ID on the img
    cv2.putText(img, str(id),
        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_TRIPLEX,
        2, (0, 255, 0), 2)
    
    # resize the img
    scale_percent = 30
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    # show the output image
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    return pixel_scale, id


def detect_leaf(image):
    '''Detect leaf in image using YOLOv8 model'''
    # model predict
    if torch.cuda.is_available():
        results = detect_model.predict(image
                            ,conf = 0.5, device = '0', save = False, save_txt = False, show_conf = False, show_labels = False)
    else:
        results = detect_model.predict(image
                            ,conf = 0.5, device = 'cpu', save = False, save_txt = False, show_conf = False, show_labels = False)
    
    
    res_plotted = results[0].plot()
    
    # resize the image
    scale_percent = 30
    width = int(res_plotted.shape[1] * scale_percent / 100)
    height = int(res_plotted.shape[0] * scale_percent / 100)
    dim = (width, height)
    res_plotted = cv2.resize(res_plotted, dim, interpolation = cv2.INTER_AREA)
    
    #show the res_plotted
    cv2.imshow("result", res_plotted)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
    return results

def counting_SOA_leaf(image,results,pixel_scale):
    '''return number of leaf and area of all leafs in image'''
    # get bbox of predict
    boxes = results[0].boxes
    
    #print number of bbox
    print(f'Number of leaf: {len(boxes)}')
    
    # get all boxes of prediction
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    SOA = 0
    
    # access each box in boxes
    for i, box in enumerate(boxes):
        # get crop of image
        crop = image[box[1]:box[3], box[0]:box[2]]
        
        # show crop image
        cv2.imshow('crop',crop)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        
        ## Blur image
        # crop = cv2.GaussianBlur(crop, (3,3),0)
        
        ## convert to hsv
        # hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        ## mask of green (36,25,25) ~ (86, 255,255)
        # mask = cv2.inRange(hsv, (25,52,72), (102, 255,255))
        
        # mask of CVPPP
        # mask = cv2.inRange(hsv, (37, 120, 120), (54, 255,255))
        
        # # mask of Lan Ho Diep
        # mask = cv2.inRange(hsv, (35, 0, 0), (55, 255,255))
        
        
        ###### Solution 1
        # # blur the mask to help remove noise
        # mask= cv2.blur(mask, (2, 2))
        # # get threshold image
        # ret, thresh = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        
        # # draw the contours on the empty image
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours = max(contours, key=lambda x: cv2.contourArea(x))
        # cv2.drawContours(crop, [contours], -1, (255, 255, 0), 2)
        # cv2.imshow("contours", crop)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        
        # count_of_pixels = cv2.contourArea(contours)
        
        
        
        ###### Solution 2
        # #show
        # cv2.imshow('mask',mask)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        
        # count_of_pixels = np.count_nonzero(mask)
        
        # # slice the green000
        # boolean_mask = mask>0
        # green = np.zeros_like(crop, np.uint8)
        # green[boolean_mask] = crop[boolean_mask]
        # #show
        # cv2.imshow('green',green)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        
        # print image size
        # print(crop.shape)
        
        
        ######### Solution 3: SAM segment
        # predictor.set_image(crop)
        # crop_height, crop_width = crop.shape[0], crop.shape[1]
        # x_center, y_center = int(crop_width/2), int(crop_height/2)
        # input_point = np.array([[x_center + 0.15*crop_width, y_center + 0.15*crop_height], [x_center - 0.15*crop_width, y_center - 0.15*crop_height], [x_center - 0.15*crop_width, y_center + 0.15*crop_height], [x_center + 0.15*crop_width, y_center - 0.15*crop_height]])
        # input_label = np.array([1, 1, 1, 1])
        # mask, scores, logits = predictor.predict(
        #     point_coords=input_point,
        #     point_labels=input_label,
        #     multimask_output=False,
        # )
        # # show segmentation mask
        # crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # plt.figure(figsize=(10,10))
        # plt.imshow(crop_rgb)
        # show_mask(mask, plt.gca())
        # show_points(input_point, input_label, plt.gca())
        # plt.title(f"Mask", fontsize=18)
        # plt.axis('off')
        # plt.show()  
        
        # h, w = mask.shape[-2:]
        # mask = mask.reshape(h, w, 1)
        # mask = mask.astype(np.uint8) * 255
        
        
        
        # # show mask
        # cv2.imshow('mask',mask)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
        
        # # # draw the contours on the empty image
        # # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # contours = max(contours, key=lambda x: cv2.contourArea(x))
        
        # # #show contours
        # # crop_copy = crop.copy()
        # # cv2.drawContours(crop_copy, [contours], -1, (255, 255, 0), 2)
        # # cv2.imshow("contours", crop_copy)
        # # cv2.waitKey()
        # # cv2.destroyAllWindows()
        
        # # count_of_pixels = cv2.contourArea(contours)
        
        # count_of_pixels = np.count_nonzero(mask)
        
        # # Print the result
        # print(f"Area of leaf {i+1}: {count_of_pixels*pixel_scale} cm^2")
        
        # SOA += count_of_pixels*pixel_scale
        
        
        ######### Solution 4: Using Yolov8 Segmentation
        # load the model from path
        segment_model = YOLO(args['segment'])  
        H, W, _ = crop.shape
        # Predict with the model
        res = segment_model(crop, conf = 0.7, verbose=False)  # predict on an image
        # Check if masks are available in the result
        if res[0].masks is not None:
            # Convert mask to numpy array
            masks = res[0].masks.data.cpu().numpy().astype(np.uint8)*255
            
            # Get the first mask
            mask = masks[0]
            mask = cv2.resize(mask, (W, H))
            
            # show mask
            cv2.imshow('mask',mask)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()
            
            # Apply the mask to the image
            segmented_img = cv2.bitwise_and(crop, crop, mask=mask)
            # show the segment
            cv2.imshow('segmented_img',segmented_img)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()
            
            count_of_pixels = np.count_nonzero(mask)
            
            # Print the result
            print(f"Area of leaf {i+1}: {count_of_pixels*pixel_scale} cm^2")
            
            SOA += count_of_pixels*pixel_scale

    print("SOA: ",SOA)
    return len(boxes), SOA

                # F:\\Leaf_Detection\\leaf_detect_dataset_tu_label\\CVPPP\\ara2013_tray14_rgb.png
                # F:\\Leaf_Detection\\leaf_detect_dataset_tu_label\\test\images\\z4257779517740_8b94dd07f8ed2bd191fe9628535d819b.jpg
                # sys.argv[2]
if __name__ == '__main__':
    if args['input'] != 0:
        # try:
            image = cv2.imread(args['input'])
            pixel_scale, id = detect_image_id(image)
            # pixel_scale = 0.444
            results = detect_leaf(image)
            number_of_leaf, SOA = counting_SOA_leaf(image,results,pixel_scale)
        # except:
            # sys.exit('No image detected. Please! try again')
    else:
        try:
            #detect leaf in photo captured by camera
            cam = cv2.VideoCapture(0)
            result, image = cam.read()
            if result:
                pixel_scale, id = detect_image_id(image)
                results = detect_leaf(image)
                number_of_leaf, SOA = counting_SOA_leaf(image,results,pixel_scale)
            else:
                sys.exit('No image detected. Please! try again')
        except:
            sys.exit('No camera detected. Please! try again')