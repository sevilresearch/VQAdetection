import cv2 as cv
import numpy as np
from time import time
import pyautogui
import imutils
from vision import Vision

cap = cv.VideoCapture('Argo.mp4')

cascade_final = cv.CascadeClassifier('cascade2/cascade.xml')

vision_limestone = Vision(None)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)
 
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

loop_time = time()
currentframe = 0
while(cap.isOpened()):

  ret, frame = cap.read()


  if ret == True:


    frame30 = rescale_frame(frame, percent=30)
    rectangles = cascade_final.detectMultiScale(frame30)
    detection_image = vision_limestone.draw_rectangles(frame30, rectangles)

    cv.imshow('Frame',frame30)

    key = cv.waitKey(1)
    if key == ord('q'):
      cv.destroyAllWindows()
      break
    elif key == ord('p'):
      name = './positive2/frame' + str(currentframe) + '.jpg'
      print ('Creating...' + name) 
  
        # writing the extracted images 
      cv.imwrite(name, frame30) 
  
        # increasing counter so that it will 
        # show how many frames are created 
      currentframe = currentframe + 1
      #cv.imwrite('positive1/{}.jpg', format(loop_time), frame30)
    elif key == ord('n'):
      name = './negative2/frame' + str(currentframe) + '.jpg'
      print ('Creating...' + name) 
  
        # writing the extracted images 
      cv.imwrite(name, frame30) 
  
        # increasing counter so that it will 
        # show how many frames are created 
      currentframe = currentframe + 1
      #cv.imwrite('negative1/{}.jpg', format(loop_time), frame30)


  # Break the loop

  else: 

    break

cap.release()
cv.destroyAllWindows()
