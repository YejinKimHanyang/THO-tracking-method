import numpy as np

MAX_LENGTH = 10

tracking_objects = {}
tracking_id = 0

class tracking_object():
    def __init__(self, tracking_id, label):
        self.id = tracking_id
        self.label = label
        self.list = []

    def append(self, obj):
        self.list.append(obj)
        if len(self.list) > MAX_LENGTH:
            self.list = self.list[1:]
       
    def check_del(self):
        if not np.any(bool(self.list)):
            return True
        
