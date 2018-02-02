import numpy as np

def shape_to_np(shape, dtype='int'):
    ''' 
    dlib face landmark detector will return the shape object containing 68 coordinates
    '''

    # initialize the list of (x,y) coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    for i in range(68):
        coords[i] = (shape.part(i)[0], shape.part(i)[1])
    
    return coords


