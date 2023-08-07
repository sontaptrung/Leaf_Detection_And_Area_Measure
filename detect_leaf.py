from ultralytics import YOLO
import cv2
import os

model = YOLO('F:\Leaf_Detection\weight\mAP_0.587_50epoch_4batch_yolov8x-p2_dataset_part1\\best.pt')  

image_train_dir = 'F:\Leaf_Detection\leaf_detect_dataset_tu_label\lcc'
list_img_path = []
list_img_path = [os.path.join(image_train_dir, x) for x in os.listdir(image_train_dir)]

# for i in range(10, 28): 
#     img_path = "F:\Leaf_Detection\leaf_detect_dataset_tu_label\CVPPP\\ara2013_tray" + str(i) + "_rgb.png"
#     list_img_path.append(img_path)

results = model.predict(list_img_path
                            ,conf = 0.8
                            , device = '0'
                            , save = False
                            , save_txt = False
                            , show_conf = False
                            , show_labels = False
                            , save_crop = True)