#values each object must have

class object():
    def __init__(self, xywh, keypoints, features):
        self.keypoints = keypoints
        self.features = features
        self.xywh = xywh
