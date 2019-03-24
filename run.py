from MotorControl import *
from server import *
from two import *
import cv2

sleep_time = 1
mc = MotorControl()
cap = cv2.VideoCapture(0)
rotate()

while(1):
	mc.forward()

	ret, frame = cap.read()
	val_y, val_x = detectObject(frame)
	print("val_y = ", val_y)
	print("val_x = ", val_x)
	#cv2.imshow('FRAM', frame)

	if val_y > 0.3:
		mc.turn_right()
		sleep(val_y)
		mc.forward()
	elif val_y < -0.3:
		mc.turn_left()
		val_y = val_y * (-1)
		sleep(val_y)
		mc.forward()	

	if val_x < 0.4 or val_x > 0.6:
		mc.stop()
		rotate()
		mc.forward()

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	'''
	sleep(sleep_time)
	rotate()
	sleep(sleep_time + 1)
	#mc.turn_left()
	#sleep(sleep_time + 1)
	rotate()
	sleep(sleep_time + 1)
	'''

cap.release()
cv2.destroyAllWindows()
