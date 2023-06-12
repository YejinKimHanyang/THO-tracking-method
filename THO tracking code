from ultralytics import YOLO
import pyrealsense2 as rs
import cv2
import numpy as np
from object_ import object
import yaml

MAX_FRAME = 20
PIXEL_DISTANCE_THRESHOLD = 200
FEATURE_MATCHING_THRESHOLD = 10

def get_model_and_labels(model_path='yolov8x.pt',
                         labels_path='./ultralytics/datasets/coco.yaml'):
    model = YOLO(model_path)
    with open(labels_path) as f:
        labels = yaml.load(f, Loader=yaml.FullLoader)['names']
    return model, labels

def inference_image(model, image):   
    results = model(image)
    #pad_w = (image.shape[1]-800)//2
    #pad_h = (image.shape[0]-600)//2
    #cv2.rectangle(image, (pad_w,pad_h), (image.shape[1]-pad_w, image.shape[0]-pad_h), (0,0,0), 2)
    #print(results)
    preds = results[0].boxes.data.cpu().numpy()
    bbox_list = preds[..., :4].astype(np.int32)
    #print(bbox_list)
    label_list = preds[..., 5].astype(np.int32)
    #print(label_list)

    bbox_list_ = []
    label_list_ = []
    for i in range(len(label_list)):
        if label_list[i] == 7: #truck
            label_list[i] = 2 #car
        elif label_list[i] == 49: #orange
            label_list[i] = 32 #sports ball
        elif label_list[i] == 47: #apple
            label_list[i] =32
        elif label_list[i] == 29: #frisbee
            label_list[i] =32
        elif label_list[i] == 45: #bowl
            label_list[i] =32

        elif label_list[i] == 40: #wine glass
            label_list[i] = 39 #bottle
        elif label_list[i] == 14: #bird 
            continue
        elif label_list[i] == 41: #cup
            continue        
        #elif label_list[i] == 0: #person
        #    continue
        elif label_list[i] == 26: 
            continue #handbag
        elif label_list[i] == 27:
            continue
        elif label_list[i] == 12:
            continue
        elif label_list[i] == 24: #backpack
            continue
        elif label_list[i] == 28: #suitcase
            continue
        elif label_list[i] == 10: #fire hydrant
            continue
        elif label_list[i] == 10: #fire hydrant
            continue

        elif label_list[i] == 11: #stop sign
            continue
        elif label_list[i] == 58: #potted plant
            continue
        bbox_list_.append(bbox_list[i])
        label_list_.append(label_list[i])

    return bbox_list_, label_list_

# draw_image(image, tracking_object.list[-1], labels)
def draw_image(image, object_, id, label, color=(255, 0, 0)):
    p1 = (object_.xywh[:2] - 0.5 * object_.xywh[2:]).astype(np.int32)
    p2 = (object_.xywh[:2] + 0.5 * object_.xywh[2:]).astype(np.int32)
    
    text = f'id:{id}, {label}'
    try:
        cv2.rectangle(image, p1, p2, (255,0,0), 2)
        cv2.putText(image, text, [p1[0], p1[1]-5], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color, 1)
    except:
        pass


def main():
    # key = None
    # items = None

    class tracking_object():
        def __init__(self, tracking_id):
            self.tracking_id = tracking_id
            # self.label = label
            # self.t_id = t_id
            self.list = []
            self.frame_id = -1
        
        def append(self, obj, frame_id):
            self.list.append(obj)
            #append meaning
            if len(self.list) > MAX_FRAME:
                self.list = self.list[1:]
            self.frame_id = frame_id

        # def id_update(self):
        #     self.frame_id += 1

        def check_del(self):
            if self.list.count(0) >= MAX_FRAME:
                print(f'{self.tracking_id} deleted')
                return True
            
        def find_last_object(self):
            for i in range(len(self.list))[::-1]:
                if self.list[i] != 0:
                    return self.list[i]
        
    #load weights model and labels
    model, labels = get_model_and_labels()
    #open video file and process inference
    # read video => bgr, realsense => rgb
    # cap = cv2.VideoCapture('./test5.avi')

    cap = cv2.VideoCapture('./output6.avi')
    # rs.pipeline()
    # pipeline = rs.pipeline()
    # pipeline.start()

    detector = cv2.ORB_create()

    frame_id = 0

    tracking_objects = [[] for _ in range(len(labels))]
    tracking_id = 1

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    writer = cv2.VideoWriter('./output6_sidewalk.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1040, 600))
    
    
    try:
        #while(True):
        while(cap.isOpened()):
            ret, frame = cap.read()


            # print(type(frame))
            #frames = pipeline.wait_for_frames()

            #corlor_frame = frames.get_color_frame()
            #frame = np.asanyarray(corlor_frame.get_data())
            
            if ret:
                # rgb

                bbox_list, label_list = inference_image(model, frame)
                #print(len(bbox_lisjt))
                #print(bbox_list)
                #print(len(label_list))
                for bbox, label in zip (bbox_list, label_list):
                    label_str = labels[label]
                    # xyxy list to x1, y1, x2, y2
                    x1, y1, x2, y2 = bbox
                    # crop image
                    crop_image = frame[y1:y2, x1:x2]
                    # extract keypoints, features
                    # bgr
                    keypoints, features = detector.detectAndCompute(crop_image, None)
                    # xyxy to xywh for centerpoint calculation in euclidean distance
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    w = (x2 - x1)
                    h = (y2 - y1)
                    xywh = np.stack([cx, cy, w, h], -1)
                    # xywh = np.concatenate([cx, cy, w, h], -1)
                    # create obejct
                    obj = object(xywh, keypoints, features)

                    # insert obj to tracking object (consider 4 steps)
                    #print('in')


                    if not tracking_objects[label]:
                        # obj.update_id(tracking_id)
                        t_obj = tracking_object(tracking_id)
                        t_obj.append(obj, frame_id)
                        #append tracking object to tracking objects list
                        #wrapped with class label
                        tracking_objects[label] += [t_obj]
                        #object list insert -> insert 0 to other list
                        tracking_id += 1

                    else:
                        min_distance = PIXEL_DISTANCE_THRESHOLD
                        best_match = -1
                        best_match_count = -1
                        for t in range(len(tracking_objects[label])):
                            ## just pixel_distance
                            t_obj = tracking_objects[label][t].find_last_object()
                            pixel_distance = np.sum(np.sqrt(np.square(t_obj.xywh[:-2] - obj.xywh[:-2])), -1)

                            #
                            if pixel_distance < min_distance: # xy distance
                                min_distance = pixel_distance
                                best_match = t
                            
                            ## with feature distance
                            else:                           
                                if np.any(t_obj.features==None) or np.any(obj.features==None):
                                    continue
                                matches = matcher.match(obj.features, t_obj.features)
                                if pixel_distance < PIXEL_DISTANCE_THRESHOLD * 2: # 2x small
                                    if len(matches) > FEATURE_MATCHING_THRESHOLD:
                                        if len(matches) > best_match_count:
                                            best_match = t
                                            best_match_count = len(matches)
                                        
                                        elif len(matches) == best_match_count:
                                            past_pixel_distance = np.sum(np.sqrt(np.square(tracking_objects[label][best_match].find_last_object().xywh[:-2] - obj.xywh[:-2])), -1)
                                            if pixel_distance < past_pixel_distance:
                                                best_match = t
                                                best_match_count = len(matches)
                        
                        if best_match == -1:
                            t_obj = tracking_object(tracking_id)
                            t_obj.append(obj, frame_id)
                            tracking_objects[label] += [t_obj]
                            tracking_id += 1
                        else:
                            tracking_objects[label][best_match].append(obj, frame_id)
                        
                        # empty_object = object()
                for key in range(len(tracking_objects)):
                    for t in range(len(tracking_objects[key])):
                        if tracking_objects[key][t].frame_id != frame_id:
                            tracking_objects[key][t].append(0, frame_id)

                # tracking objects for loop
                for key in range(len(tracking_objects)):
                    t=0
                    while(True):
                        if t >= len(tracking_objects[key]):
                            break
                        if tracking_objects[key][t].check_del():
                            del tracking_objects[key][t]
                            continue
                        t += 1
                            
                for key in range(len(tracking_objects)):
                    for t in range(len(tracking_objects[key])):
                        try:
                            draw_image(frame, tracking_objects[key][t].find_last_object(), tracking_objects[key][t].tracking_id, labels[key])
                        except KeyError:
                            print(tracking_objects[key], 1)
                frame_id += 1
                cv2.imshow('frame', frame)
                #[..., ::-1]
                #if cv2.waitKey(30) == ord('q'):
            if cv2.waitKey(1) == 27:
                break

                #video write
            #[..., ::-1]
            print(frame)
            writer.write(frame)      
            # print(type(frame)) 
    finally:
        writer.release()
        cap.release()
        #pipeline.stop()
        cv2.destroyAllWindows()

main()
