#!/bin/env python3
import numpy as np
import cv2

cap = cv2.VideoCapture(0) #("/dev/video0")

while (True):
	ret, frame =cap.read()
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	print(frame.shape)
	cv2.imshow('FRAM', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
