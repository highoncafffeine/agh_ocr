import numpy as np
import cv2
import os
import pytesseract

def deskew(image, angle):
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
custom_config = "--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

image_directory = 'Spandan'
image_name = 'REFERENCE_IMAGE7.jpg'# 'REFERENCE_IMAGE7.jpg'
image_path = os.path.join(image_directory,image_name)

img = cv2.imread(image_path)
# img = cv2.resize(img, None, fx=0.3, fy=0.3)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
font = cv2.FONT_HERSHEY_COMPLEX

ret, thresh = cv2.threshold(imgray, 200, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

coords = []
img1 = img.copy()
if len(contours) != 0:
     for contour in contours:
         _, (w,h), angle = cv2.minAreaRect(contour)
         area = cv2.contourArea(contour)
         if area > 15000 and (w*h-area)<0.15*w*h :
             x, y, w, h = cv2.boundingRect(contour)
             coords += [(x,y,w,h)]
             rect = cv2.minAreaRect(contour)
             box = cv2.boxPoints(rect)
             box = np.int0(box)
             cv2.drawContours(img,[contour],0,(255,0,0),3)
             cv2.drawContours(img,[box],0,(0,0,255),3)

img_small = cv2.resize(img, None, fx=0.3, fy=0.3)
cv2.imshow('Image', img_small)
cv2.waitKey(0)
img1 = cv2.resize(img1, None, fx=0.3, fy=0.3)
srno = []
for x,y,w,h in coords:
    x, y, w, h = 3*x//10,3*y//10,3*w//10,3*h//10
    img0 = img1[y:y+h,x:x+w]
    # Preprocessing before detection
    img_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 200, 255, 0)
    # cv2.imshow('Thresh',thresh)

    # Detecting the words
    hImg, wImg, _ = img0.shape
    boxes = pytesseract.image_to_data(thresh, config = custom_config)
    
    for x, b in enumerate(boxes.splitlines()):
        if x != 0:
            b = b.split()
            # print(b)
            if len(b) == 12:
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                area = 0
                if len(b[11])>4:
                    print(b[11])
                    cv2.rectangle(img0, (x, y), (w+x, h+y), (0, 0, 255), 3)
                    cv2.putText(img0, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 50, 255), 2)
                    srno += [b[11]]


    cv2.imshow('Result', img0)

    cv2.waitKey(0) # putting a delay to the display window

    cv2.destroyAllWindows()
# if(len(srno)==1):
#     image_old = os.path.join(image_directory,image_name)
#     ext = os.path.splitext(image_name)[1]
#     image_new = os.path.join(image_directory,srno[0]+ext)
#     os.rename(image_old,image_new)
# else:
#     pass