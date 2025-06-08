import os
#from IPython import display
import matplotlib.pyplot as plt
from PIL import Image
import ultralytics
from ultralytics import YOLO
import cv2
#from ultralyticsplus import YOLO, render_result      
#from yolov8 import YOLO
import numpy as np
import glob
import torch
import torchvision
from torchvision import transforms as T



model = YOLO("C:/Users/nada/OneDrive/Desktop/masked_detection_usingyolov8/unmasked face segmentation/yolov8n-seg.pt") 
model.train(data="C:/Users/nada/OneDrive/Desktop/masked_detection_usingyolov8/unmasked face segmentation/data_singleclass.yaml", epochs=5) 
model.val(data="C:/Users/nada/OneDrive/Desktop/masked_detection_usingyolov8/unmasked face segmentation/data_singleclass.yaml")
#results = model('C:/Users/nada/OneDrive/Desktop/masked_detection_usingyolov8/unmasked face segmentation/img.jpg') 

confusion_matrix = cv2.imread("C:/Users/nada/OneDrive/Desktop/masked_detection_usingyolov8/unmasked face segmentation/runs/segment/train/confusion_matrix.png")
confusion_matrix_resized = cv2.resize(confusion_matrix, (1500, 700)) 
cv2.imshow("Confusion Matrix", confusion_matrix_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

model = YOLO(model="C:/Users/nada/OneDrive/Desktop/masked_detection_usingyolov8/unmasked face segmentation/runs/segment/train/weights/best.pt")
test_path = "C:/Users/nada/OneDrive/Desktop/masked_detection_usingyolov8/unmasked face segmentation/dataset_singleclass/test/images"
filenames = glob.glob(test_path + '/*.png', recursive=False) + glob.glob(test_path + '/*.jpg', recursive=False)
test_image1 = cv2.imread(filenames[0])
results = model.predict([ test_image1], save=True, line_thickness=1)
predicted_image_path = "runs/segment/predict/image0.jpg"
predicted_image=cv2.imread(predicted_image_path)
cv2.imshow("predicted image",predicted_image)
#cv2.imshow("predicted",test_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

def parse_yolo_format(txt_path, image_width, image_height):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
    class_id, *coords = map(float, lines[0].split())
    coords = [int(coord * image_width) if i % 2 == 0 else 
              int(coord * image_height) for i, coord in enumerate(coords)]  
    return class_id, coords



def cut_roi_save_transparent(image_path, txt_path, save_path):
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_height, image_width, channels = original_image.shape
    
    if channels == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
    
    class_id, coords = parse_yolo_format(txt_path, image_width, image_height)
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    pts = np.array(coords, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], color=255)
    
    # Apply mask to the image to keep only the ROI
    roi = cv2.bitwise_and(original_image, original_image, mask=mask)
    
    transparent_img = np.zeros((image_height, image_width, 4), dtype=np.uint8)
    
    x, y, w, h = cv2.boundingRect(mask)
    transparent_img[y:y+h, x:x+w] = roi[y:y+h, x:x+w]
    
    cv2.imwrite(save_path, transparent_img)
    


    cv2.imshow('Cropped ROI', transparent_img)
    cv2.imshow("predicted image",predicted_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = "C:/Users/nada/OneDrive/Desktop/masked_detection_usingyolov8/unmasked face segmentation/runs/segment/predict/image0.jpg"
txt_path = "C:/Users/nada/OneDrive/Desktop/masked_detection_usingyolov8/unmasked face segmentation/dataset_singleclass/test/labels/with-mask.txt"
save_path = "C:/Users/nada/OneDrive/Desktop/masked_detection_usingyolov8/unmasked face segmentation/runs/segment/predict/segmented_region.png"

cut_roi_save_transparent(image_path, txt_path, save_path)

