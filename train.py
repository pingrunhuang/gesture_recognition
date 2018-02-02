import cvutils
import pickle
from GestureException import NoKeypointsException, NotEnoughKeypointsException

from sklearn.cluster import KMeans
from sklearn import svm

def get_data_path():
    import os
    data_dir='./dataset/'
    posture_dirs=os.listdir(data_dir)
    data_path_dict={}
    for d in posture_dirs:
        data_path_dict[d] = [data_dir+d+"/"+x for x in os.listdir(data_dir+d)]
    return data_path_dict

def keypoints_clustering(keypoints, cluster_num=8):
    try:
        if keypoints is None or len(keypoints)==0:
            raise NoKeypointsException()
        if len(keypoints)<cluster_num:
            raise NotEnoughKeypointsException(len(keypoints), cluster_num)
        kmeans=KMeans(n_clusters=cluster_num).fit(keypoints)
        return kmeans

    except NoKeypointsException as e:
        # print(e.msg)
        return
    except NotEnoughKeypointsException as e:
        # print(e.msg)
        return

def save_to_pickle(gesturename, obj):
    with open('./kmeans_result/{}.pickle'.format(posture_name), mode='wb') as f:
        pickle.dumps(obj, f)

def save_to_txt(gesturename):
    pass

def train():
    pass

if __name__=='__main__':
    data_dict=get_data_path()
    training_features = {}
    
    # for gesture_name in data_dict:
    #     training_features[gesture_name]=[]
    #     for path in data_dict[gesture_name]:
    #         img = cvutils.read_image_from_dir(path)
    #         keypoints, descriptors=cvutils.retrieve_keypoints(img)
    #         kmeans = keypoints_clustering(descriptors)
    #         if kmeans is not None:
    #             training_features[gesture_name].append(list(kmeans.cluster_centers_))

    # with open('training_features.pickle', mode='wb') as f:
    #     pickle.dump(training_features, f)

    with open('training_features.pickle', mode='rb') as f:
        training_features = pickle.load(f)
    labels = [x for x in training_features.keys()]
    print(labels)
    points = training_features['Five'][0]
    print(points[0])
    clssifier1 = svm.SVC()
    clssifier1.fit(points, [1])

    


            