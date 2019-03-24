import cv2
import numpy as np


def detectObject(img):
    #file_Path = "./train.jpg"

    #img = cv2.imread(file_Path, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[:,0:3024]
    img = cv2.resize(img, (320,320))
    imgShape = img.shape
    finalPosx = 160
    finalPosy = 160
    # print(imgShape)
    count = 0
    set = False
    for i in range(imgShape[0]):
        xArr = img[i,:]
        fx = np.array(xArr, dtype = float)
        fx = np.gradient(fx)
        for j in range(imgShape[1]):
          if img[i,j] > abs(20):
              if count >= 30:
                  print("done!")
                  finalPosx = i
                  finalPosy = j
                  set = True
                  break

              count = 0
              continue
              #first = True
          else:
              count+=1
        if set:
            print(finalPosx)
            print(finalPosy)
            break

    returnVal = finalPosy/320
    return returnVal, finalPosx/320
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
