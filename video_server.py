import cv2
import numpy as np
from socket import *

HOST = '127.0.0.1'  
PORT = 1024  
  
s = socket(AF_INET,SOCK_DGRAM)  
s.bind((HOST,PORT))  
print('...waiting for message..')
while True:  
	data, address = s.recvfrom(65536 * 1024)  
	data = np.fromstring(data, np.uint8)
	img = cv2.imdecode(data, 1)
	cv2.imshow("img_decode", img)
	if cv2.waitKey(1) == 27:
		break
s.close()
cv2.destroyAllWindows()
