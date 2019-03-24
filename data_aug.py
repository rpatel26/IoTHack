import cv2
import os
import glob

file_Path = './data/paper/'
file_Name = 'paper'

num_Files = len([name for name in os.listdir(file_Path)])
print(num_Files)
for i in range(1, num_Files):
	file_Temp = '{}{}{}.jpg'.format(file_Path, file_Name, i)
	if (os.path.isfile(file_Temp)):
		print(file_Temp)
		cv2.imread(file_Temp)
		flipped_Image =  cv2.flip(cv2.imread(file_Temp), -1 )
		cv2.imwrite('{}{}{}_flip.jpg'.format(file_Path, file_Name, i), flipped_Image)

