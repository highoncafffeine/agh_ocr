import numpy as np
import cv2
import os
from easyocr import Reader


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
	boxes = pytesseract.image_to_data(thresh, config = custom_config)

	for x, b in enumerate(boxes.splitlines()):
		if x != 0:
			b = b.split()
			# print(b)
			if len(b) == 12:
				print(b[11])
	# img1 = cv2.resize(img1, None, fx=0.3, fy=0.3)
	srno = []
	for x,y,w,h in coords:
		
		fx = 0.3
		img0 = img1[y:y+h,x:x+w]
		cv2.imshow("thresh",thresh)
		cv2.waitKey(0)
		langs = ["en"]
		reader = Reader(langs, gpu=True)
		results = reader.readtext(img0)
		for (bbox, text, prob) in results:
			if(len(text) > 5):
				# display the OCR'd text and associated probability
				print("[INFO] {:.4f}: {}".format(prob, text))
				# unpack the bounding box
				(tl, tr, br, bl) = bbox
				tl = (int(tl[0]), int(tl[1]))
				tr = (int(tr[0]), int(tr[1]))
				br = (int(br[0]), int(br[1]))
				bl = (int(bl[0]), int(bl[1]))
				# cleanup the text and draw the box surrounding the text along
				# with the OCR'd text itself
				# text = cleanup_text(text)
				cv2.rectangle(img0, tl, br, (0, 255, 0), 2)
				cv2.putText(img0, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		# show the output image
				cv2.imshow(img0)
				cv2.waitKey(0) 
		cv2.destroyAllWindows()
		# for x, b in enumerate(boxes.splitlines()):
		# 	if x != 0:
		# 		b = b.split()
		# 		# print(b)
		# 		if len(b) == 12:
		# 			x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
		# 			area = 0
		# 			if len(b[11])>4:
		# 				print(b[11])
		# 				cv2.rectangle(img0, (x, y), (w+x, h+y), (0, 0, 255), 3)
		# 				cv2.putText(img0, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 50, 255), 2)
		# 				srno += [b[11]]


		# 				cv2.imshow('Result', img0)

		# 				cv2.waitKey(0) # putting a delay to the display window

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