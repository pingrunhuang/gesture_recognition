import cv2
import cvutils

'''
The mac front cam resolution is 720 x 1280 
'''

if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # convert to gray
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints=cvutils.retrieve_keypoints(frame)
        
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()