import numpy as np
import cv2

# instanciate camera object to capture images
cam = cv2.VideoCapture(0)

# creating a haar-cascade classifier for face detection
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# container to store data
data = []
idx = 0		# current frame number

while True:
	# retrieves the status(type=boolean) and frame from camera
	ret, frame = cam.read()

	#if frame is received from camera
	if ret == True:
		# convert the given frame to grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# apply the haar cascade to detect faces in the current frame
		# other 2 parameters 1.3 and 5 are fine tuning parameters for 
		# haar cascade object
		faces = face_cas.detectMultiScale(gray, 1.3, 5)

		# for each face we have
		# x,y i.e. coordinate of left bottom point of face box
		# w and h are height and width of face box
		for (x,y,w,h) in faces:
			# extract face component from image
			face_component = frame[y:y+h,x:x+w, :]
			
			# resize the face image to 50*50*3
			fc = cv2.resize(face_component, (50,50))
			# store 60 the face image after every 10 frames 
			if idx%10 == 0 and len(data) < 60:
				data.append(fc)

			# for visualisation, draw a rectange around the face
			# in the image
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
		idx += 1	# increment the frame number
		cv2.imshow('frame',frame)	# display the frame

		# if the user presses ESCAPE key or 60 frames are taken.. break
		if cv2.waitKey(1) == 27 or len(data) >= 60:
			break

	else:
		print "Error Capturing Photo"
		break

# destroy all cv windows
cv2.destroyAllWindows()

# convert the data to numpy format
data = np.asarray(data)

print data.shape

# save the data in a file as numpy matrix in an encoded format
np.save('face_01', data)	

# we run this program for each person and store their face data in
# seperate files

