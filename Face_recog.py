import os
import cv2
import face_recognition
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import math
import os
import time
from pygame import mixer

print(os.getcwd())
img_path = os.getcwd() + "//FaceRecogDataset"

images = []
class_names = []
encode_list = []
encode_list_cl = []
myList = os.listdir(img_path)

print(myList)

for subdir in os.listdir(img_path):
    path = img_path + '/' + subdir
    path = path + '/'
    for img in os.listdir(path):
        img_pic = path + img
        class_names.append(subdir)
        cur_img = cv2.imread(img_pic)
        cur_img = cv2.cvtColor(cur_img , cv2.COLOR_BGR2RGB)
        images.append(cur_img)

def detect_and_predict_mask(frame, faceNet, maskNet,threshold):
	# grab the dimensions of the frame and then construct a blob from it
	global detections 
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces locations,
	# and the list of predictions from our face mask network
	locs = []
	preds = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence >threshold:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
            
			# add the face and bounding boxes to their listqqs
			locs.append((startX, startY, endX, endY))
			preds.append(maskNet.predict(face)[0][0])
	return (locs, preds)
# SETTINGS
MASK_MODEL_PATH=os.getcwd()+"//masksdetection//model//model.h5"
FACE_MODEL_PATH=os.getcwd()+"//masksdetection//face_detector"
SOUND_PATH=os.getcwd()+"//masksdetection//sounds//alarm.wav" 
THRESHOLD = 0.5

# Load Sounds
mixer.init()
sound = mixer.Sound(SOUND_PATH)

protoPath = os.getcwd()+"//masksdetection//face_detector//deploy.prototxt"
weightsPath =os.getcwd()+"//masksdetection//face_detector//res10_300x300_ssd_iter_140000.caffemodel"
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
faceNet = cv2.dnn.readNet(protoPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(MASK_MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(0).start()
time.sleep(1.0)
        
def find_encodings(images):
    #for names in images :
	for img in images:
		face_locations = face_recognition.face_locations(img)
		if len(face_locations) > 0:
			encodings = face_recognition.face_encodings(img, face_locations)[0]
			encode_list.append(encodings)
	return encode_list    
encodeListKnown = find_encodings(images) 

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	original_frame = frame.copy()
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet,THRESHOLD)
	facesCurFrame = face_recognition.face_locations(frame)
	encodeCurFrame  = face_recognition.face_encodings(frame,facesCurFrame)
	print("pred: ",preds)
    
	for encodeFace, faceLoc, pred in zip(encodeCurFrame, facesCurFrame, preds):
		# (mask, withoutMask) = pred
		(startY, endX, endY,startX) = faceLoc
		if pred < 1:
			label = "No Mask"
			color = (0, 0, 255)
			cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.rectangle(frame, (startX, startY+math.floor((endY-startY)/1.6)), (endX, endY), color, -1)

			sound.play()
			while mixer.music.get_busy():
				pass				
		else:			
			label = "Mask"
			color = (0, 255, 0)	
			cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.rectangle(frame, (startX, startY+math.floor((endY-startY)/1.6)), (endX, endY), color, -1)

		label = "{}: {:.2f}%".format(label, pred * 100)
		matches = face_recognition.compare_faces(encode_list, encodeFace)
		faceDis = face_recognition.face_distance(encode_list, encodeFace)
		matchIndex = np.argmin(faceDis)
		name = class_names[matchIndex]

		(startY, endX, endY,startX) = faceLoc
		cv2.putText(frame, name, (startX, startY - 23),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2 )
		# cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		# cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		# cv2.rectangle(frame, (startX, startY+math.floor((endY-startY)/1.6)), (endX, endY), color, -1)

	cv2.addWeighted(frame, 0.5, original_frame, 0.5 , 0,frame)
	frame= cv2.resize(frame,(860,490))
	cv2.imshow("Masks Detection", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()     
    
    
