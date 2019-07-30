import numpy as np
import cv2

cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX


f_01 = np.load('face_01.npy').reshape((60,50*50*3))
f_02 = np.load('face_02.npy').reshape((60,50*50*3))
f_03 = np.load('face_03.npy').reshape((60,50*50*3))

print f_01.shape

names = {
	0 : 'Deepak',
	1 : 'Kaustubh',
	2 : 'Krishna',
}

labels = np.zeros((180,1))
labels[:60, :] = 0.0
labels[60:120, :] = 1.0
labels[120:, :] = 2.0

data = np.concatenate([f_01,f_02,f_03])

print data.shape


def distance(x, y):
	return np.sqrt(((x-y)**2).sum())


def knn(point, train, targets, k=5):
	m = train.shape[0]
	dist = []
	for x in range(m):
		#computing distance of test point from each train example
		dist.append(distance(point,train[x]))
	dist = np.asarray(dist)
	indx = np.argsort(dist)
    
    #Note that although we get indexes sorted based on distances but argsort does not modify dist at all
    
	sorted_labels = labels[indx][:k]
	counts = np.unique(sorted_labels,return_counts=True)
	#print counts
	print "Classified to class = ", counts[0][np.argmax(counts[1])]
	# print "returned ", 
	return counts[0][np.argmax(counts[1])]


while True:
	ret, frame = cam.read()

	if ret == True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cas.detectMultiScale(gray, 1.3, 5)

		for (x,y,w,h) in faces:
			face_component = frame[y:y+h,x:x+w, :]
			fc = cv2.resize(face_component, (50,50))
			
			lab = knn(fc.flatten(), data, labels)
			text = names[int(lab)]
			cv2.putText(frame, text, (x,y), font, 1, (255,255,0), 2)
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)

		cv2.imshow('frame', frame)
		# cv2.waitKey(0)
		# print "here---------\n"

		if cv2.waitKey(1) == 27:
			break
	else:
		print "Error"
		break

cv2.destroyAllWindows()


