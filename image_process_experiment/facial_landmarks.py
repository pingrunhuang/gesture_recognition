from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

arguments = argparse.ArgumentParser()
arguments.add_argument('-p', '--shape-predictor', required=True, help='path to facial landmark predictor')
arguments.add_argument('-i', '--image', required=True, help='path to input image')
# vars is equivalent to a dict
args = vars(arguments.parse_args())


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

image = cv2.imread(args['image'])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect face
# The second parameter is the number of image pyramid layers to apply when upscaling the image prior to applying the detector 
# to increase the reolution of the input image
rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # convert the dlib's rectangle to a opencv style bounding box 
    (x,y,w,h)=face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x,y), (x+w, y+h), color=(0,255,0), thickness=2)
    # show the face number
    cv2.putText(image, "Face #{}".format(i), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)

    # draw the landmarks coordinates on the image
    for (x, y) in shape:
        cv2.circle(image, center=(x,y), radius=1,color=(0,0,255),thickness=1)

cv2.imshow("output", image)
cv2.waitKey(0)
