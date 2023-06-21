from ultralytics import YOLO
import cv2
import numpy as np
#load model


model = YOLO('yolov8x.pt')
writer = cv2.VideoWriter('./output6_yolov8.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1040, 600))

# print(type(writer))

#exempt label list as in edited_code.py
#bird, cup, person out
#orange, apple, frisbee, bowl -> sports ball
#wine glass -> bottle



#track with model
results = model.track('./output1.avi', show=True, tracker= 'bytetrack.yaml')
# ret = np.array(results)

# # print(results)
# for result in results:
#     writer.write(results)
#     for k in result.keys:
#         print(getattr(result, k).numpy())
#     break

#     # writer.write(result.orig_img)
    
    # if cv2.waitKey(1) == 27:
    #     break
    # break
# print(type(ret))

while(True):
    writer.write(results)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
writer.release()
