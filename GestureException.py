class NoKeypointsException(Exception):
    def __init__(self):
        self.msg="No keypoints get extracted! Ignoring..."
        Exception.__init__(self, self.msg)

class NotEnoughKeypointsException(Exception):
    def __init__(self, numberOfKeypoints, numberOfClusters):
        self.msg="Number of keypoints {} should be larger then number of cluster {}! Ignoring...".format(numberOfKeypoints, numberOfClusters)
        self.keypoint_num=numberOfKeypoints
        self.cluster_num=numberOfClusters
        Exception.__init__(self, self.msg)
