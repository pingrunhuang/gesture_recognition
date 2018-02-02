import cv2
import numpy as np

'''
This module are some common methods that I used in this project

Dataset description:
        rows    cols
C:      70-90   60-70
'''

def generate_image(red, green, blue):
    '''
    :input: red, green and blue value for each pixel in the image
    :return: a numpy array with 3 dimensions
    '''

    cols=720
    rows=720

    img = np.zeros([rows,cols,3])
    img[:,:,0]=np.ones([rows,cols])*red
    img[:,:,1]=np.ones([rows,cols])*green
    img[:,:,2]=np.ones([rows,cols])*blue

    cv2.imwrite('test.jpg',img)
    return img

def retrieve_keypoints(img, is_output_keypoints=False,output='img_with_keypoints.jpg'):
    '''
    Since the SIFT algo for feature extration is a non free algorithm, therefore I choose ORB as an alternative
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.370.4395&rep=rep1&type=pdf
    '''
    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)
    keypoints=orb.detect(img, None)
    # compute the descriptors with ORB
    keypoints, descriptors = orb.compute(img, keypoints)
    # print("Number of keypoints:", len(keypoints))
    # Here kp will be a list of keypoints and des is a numpy array of shape Number_of_Keypoints×128.
    # print("Feature size: ", descriptors)
    # draw only keypoints location,not size and orientation
    '''
    DEFAULT = 0, // Output image matrix will be created (Mat::create),
                     // i.e. existing memory of output image may be reused.
                     // Two source images, matches, and single keypoints
                     // will be drawn.
                     // For each keypoint, only the center point will be
                     // drawn (without a circle around the keypoint with the
                     // keypoint size and orientation).
        DRAW_OVER_OUTIMG = 1, // Output image matrix will not be
                       // created (using Mat::create). Matches will be drawn
                       // on existing content of output image.
        NOT_DRAW_SINGLE_POINTS = 2, // Single keypoints will not be drawn.
        DRAW_RICH_KEYPOINTS = 4 // For each keypoint, the circle around
                       // keypoint with keypoint size and orientation will
                       // be drawn.
    '''
    cv2.drawKeypoints(img,keypoints,color=(0,255,0), outImage=img, flags=1)
    if is_output_keypoints:
        cv2.imwrite(output, img)

    return keypoints, descriptors

def read_image_from_dir(img_dir):
    return cv2.imread(img_dir)

def show_img_rows(img):
    print(len(img))

def show_img_cols(img):
    print(len(img[0]))

def convert_img_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

if __name__=='__main__':
    img=read_image_from_dir('./dataset/Five/Five-train116.ppm')
    img=convert_img_to_gray(img)
    keypoints=retrieve_keypoints(img, is_output_keypoints=True)
    print(len(keypoints))