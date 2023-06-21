'''
            #append frame in frames
            #fs.add(f)
            frame_count += 1

            #compare between objects within current frame with the previous frame's object
            #TODO:compare between current object's centerpoint and previous object's centerpoint

            #--------------- search function within the frames' frame list
            def search_frame(frames, frame_count):
                #how to know the index of the frame within frames list?            
                for i, element in enumerate(frames):
                    if i == 0:
                        continue
                    if i == (frame_count):
                        #finding previous frame from frames list
                        previous_element = frames[i-1]
                        return previous_element
            #-----------
'''


#Track Management algorithm
            #initial setting
            for i in range(len(f.objects_lists)):
                if len(f.objects_lists) == 0 and len(tracking_objects) == 0:
                    continue
                if len(f.objects_lists) > 0 and len(tracking_objects) == 0:
                    for i in range(len(f.objects_lists)):
                        #create tracking object instance of the specific id and insert into dictionary
                        t_o= tracking_object(f[i])
                        tracking_objects[t_o.class_type] = t_o
                        #check the current tracking stance with previous frame objects if it exists
                if len(f.objects_lists) == 0 and len(tracking_objects) > 0:
                    #append None
                    for key in tracking_objects:
                        tracking_objects.setdefault(key, []).append(None)
                #compare current frame with the previous frames
                if len(f.objects_lists) > 0 and len(tracking_objects) > 0:
                    for j in f.objects_lists[j]:
                        for key in tracking_objects:
                            #check if tracking_objects have same class type
                            if f.objects_lists[j].class_type == key:
                                for value in tracking_objects[key]:
                                    #check if within distance threshold
                                    temp_cp = f.objects_lists[j].centerpoint
                                    temp_cpprev = tracking_objects[key][value].centerpoint
                                    #both have tuple for (x,y)
                                    distance = ((temp_cp[0]-temp_cpprev[0])**2 + (temp_cp[1]-temp_cpprev[1])**2)
                                    if distance > 10:
                                        continue
                                    else:
                                        #if key/feature similar - use bf matching algorithm
                                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                                        matches = matcher.match(f.objects_lists[j].features, tracking_objects[key][value].features, crossCheck=True)
                                        if len(matches) > 50:
                                            continue
                                        else:
                                            #consider to be the same object
                                            tracking_objects.setdefault(key, []).append(f.objects_lists[j])
                                        #has the same class and close distance but different features=> set new key-value
                                        t_o = tracking_object(f.objects_lists[j])
                                        tracking_objects[t_o.class_type] = t_o
                                    #set same id
                                    #update r_id

                                    #has same class but distance too far=> set new key-value
                                    t_o = tracking_object(f.objects_lists[j])
                                    tracking_objects[t_o.class_type] = t_o
                            #does not match any prevalent class
                            else:
                                t_o= tracking_object(f.objects_lists[j])
                                tracking_objects[t_o.class_type] = t_o
                else:
                    continue



            #draw bounding box and id of the tracked object



            if cv2.waitKey(30) == ord('q'):
                break
            #draw_image(frame, *inference_result, labels)

            #cv2.imshow('frame', frame)
            #if cv2.waitKey(30) == ord('q'):
            #    break
        
    cap.release()
    cv2.destroyAllWindows()




                
        ''' for obj in objects:
                for t_obj in tracking_objects:
                    # class match {key}: label
                    # distance compute and find minimum [index]: idx
                    # minmum distance t_obj {label}[index]
                t_obj{label}[index].append(obj)

            # append x 
            t_obj{label}[index].append(0)
            t_obj{label}[index].check_del()

        '''   



                
            # print(bbox_list) 
            # #get all the bbox list of the current frame and crop
            # crop_list = []
            
            # if len(bbox_list) == 0:
            #     continue
            
            # #if it contains bounding box info about at least 1 object
            # if len(bbox_list) >= 1:
            #     len_bbox_list = len(bbox_list)
            #     #print(bbox_list[0])
            #     for i in range(len_bbox_list):
            #         #check how it crops
            #         #cv2.imshow('cropped image', crop_image(frame, bbox_list[i]))
            #         #why can't it save all the outcome?
            #         crop_list.append(crop_image(frame, bbox_list[i]))
            #         #cv2.imwrite(f'./output_image{i}.jpg', crop_image(frame, bbox_list[i]))

            # #print(crop_list)
            # #calculate keypoints of crop_list
            # orb = cv2.ORB_create()
            # keypoints_list = []
            # features_list = []
            # for i in range(len(crop_list)):
            #     keypoints, descriptors = orb.detectAndCompute(crop_list[i], None)
            #     keypoints_list.append(keypoints)
            #     features_list.append(descriptors)

            # #calculate centerpoint of the crop_list
            # centerpoint_list = []
            # #first convert to gray scale
            # for i in range(len(crop_list)):
            #     gray_img = cv2.cvtColor(crop_list[i], cv2.COLOR_BGR2GRAY)
            #     moment = cv2.moments(gray_img)
            #     x = int(moment["m10"]/moment["m00"])
            #     y = int(moment["m01"]/moment["m00"])
            #     tuple = (x,y)
            #     centerpoint_list.append(tuple)

            # #set all the values of the object instance
            # for i in range(len(crop_list)):
            #     #create object instance
            #     ob = object(keypoints_list[i], features_list[i], centerpoint_list[i], inference_result[2])





            #     #add to frame_ instance
            #     f.add(ob)

            # #Track Management algorithm
            # #initial setting
            # for i in range(len(f.objects_lists)):
            #     if len(f.objects_lists) == 0 and len(tracking_objects) == 0:
            #         continue
            #     if len(f.objects_lists) > 0 and len(tracking_objects) == 0:
            #         for i in range(len(f.objects_lists)):
            #             #create tracking object instance of the specific id and insert into dictionary
            #             t_o= tracking_object(f[i])
            #             tracking_objects[t_o.class_type] = t_o
            #             #check the current tracking stance with previous frame objects if it exists
            #     if len(f.objects_lists) == 0 and len(tracking_objects) > 0:
            #         #append None
            #         for key in tracking_objects:
            #             tracking_objects.setdefault(key, []).append(None)
            #     #compare current frame with the previous frames
            #     if len(f.objects_lists) > 0 and len(tracking_objects) > 0:
            #         for j in f.objects_lists[j]:
            #             for key in tracking_objects:
            #                 #check if tracking_objects have same class type
            #                 if f.objects_lists[j].class_type == key:
            #                     for value in tracking_objects[key]:
            #                         #check if within distance threshold
            #                         temp_cp = f.objects_lists[j].centerpoint
            #                         temp_cpprev = tracking_objects[key][value].centerpoint
            #                         #both have tuple for (x,y)
            #                         distance = ((temp_cp[0]-temp_cpprev[0])**2 + (temp_cp[1]-temp_cpprev[1])**2)
            #                         if distance > 10:
            #                             continue
            #                         else:
            #                             #if key/feature similar - use bf matching algorithm
            #                             matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            #                             matches = matcher.match(f.objects_lists[j].features, tracking_objects[key][value].features, crossCheck=True)
            #                             if len(matches) > 50:
            #                                 continue
            #                             else:
            #                                 #consider to be the same object
            #                                 tracking_objects.setdefault(key, []).append(f.objects_lists[j])
            #                             #has the same class and close distance but different features=> set new key-value
            #                             t_o = tracking_object(f.objects_lists[j])
            #                             tracking_objects[t_o.class_type] = t_o
            #                         #set same id
            #                         #update r_id

            #                         #has same class but distance too far=> set new key-value
            #                         t_o = tracking_object(f.objects_lists[j])
            #                         tracking_objects[t_o.class_type] = t_o
            #                 #does not match any prevalent class
            #                 else:
            #                     t_o= tracking_object(f.objects_lists[j])
            #                     tracking_objects[t_o.class_type] = t_o
            #     else:
            #         continue



            #draw bounding box and id of the tracked object



    #     if cv2.waitKey(30) == ord('q'):
    #         break
    #         #draw_image(frame, *inference_result, labels)

    #         #cv2.imshow('frame', frame)
    #         #if cv2.waitKey(30) == ord('q'):
    #         #    break
        
    # cap.release()
    # cv2.destroyAllWindows()





