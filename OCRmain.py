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

def Otsu(image):
    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    # if is_normalized:
    #     hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold+20

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
custom_config = "--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

sr_dict = {}
image_directory = 'Spandan'
for image_name in os.listdir(image_directory):
    # image_name = 'image'+str(i)+'.jpg'
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
            if area > 15000 and (w*h-area)<0.5*w*h :
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
        # fx = 400/w
        # print(fx)
        
        fx = 0.3
        x, y, w, h = int(x*fx), int(y*fx), int(w*fx), int(h*fx)
        img0 = img1[y:y+h,x:x+w]
        
        # img0 = cv2.resize(img0, None, fx=fx, fy=fx, interpolation=cv2.INTER_AREA)
        # # Preprocessing before detection
        img_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 200, 255, 0)
        # # cv2.imshow('Thresh',thresh)
        # # Sharpen Image
        # kernel = np.array([[0, -1, 0],
        #             [-1, 5,-1],
        #             [0, -1, 0]])
        # thresh = cv2.filter2D(src=thresh, ddepth=-1, kernel=kernel)
        # # thresh = cv2.erode(thresh, np.ones((5,5),np.uint8), iterations=1)
        # # thresh = cv2.dilate(thresh, np.ones((5,5),np.uint8), iterations=1)
        # # Detecting the words
        # hImg, wImg, _ = img0.shape
        # 
        # boxes = pytesseract.image_to_data(thresh, config = custom_config)
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
    if(len(srno)==1):
        sr_dict[image_name] = srno
        # image_old = os.path.join(image_directory,image_name)
        # ext = os.path.splitext(image_name)[1]
        # image_new = os.path.join(image_directory,srno[0]+ext)
        # os.rename(image_old,image_new)
    else:
        sr_dict[image_name] = srno
        pass
    print()