import numpy as np
import cv2
import sys
import json
import time

def open_json(path, name):
    with open(path + name, 'r') as f:
        ret_dict = json.loads(f.read())
    return ret_dict

if __name__ == '__main__' :
    cap = cv2.VideoCapture("server/network/testfiles/testvideo.mpg")
    video_data = open_json("server/metrics/storage/sessions/riverrun_test/kerasmon/","data_video_test.json")
    time.sleep(2)
    if not cap.isOpened():
        print("Could not open video")
        sys.exit()
    tracker = cv2.TrackerCSRT_create()
    # Initialize tracker with first frame and bounding box
    predictions = video_data["predictions"]
    positive_index = video_data["class_indices"]["positive"]
    count = 0
    tracker_init = False
    while(cap.isOpened()):
        #print("Cap opened")
        ok, frame = cap.read()
        if not ok:
            break
        if predictions[count][positive_index] > 0.5:
            #print("Positive detection!")
            cv2.putText(frame, "Detection!", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            if not tracker_init:
                tracker = cv2.TrackerCSRT_create()
                bbox = cv2.selectROI(frame, False)
                if bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0:
                    print("Nothing selected")
                else:
                    ok = tracker.init(frame, bbox)
                    tracker_init = ok
            else:
                ok, bbox = tracker.update(frame)
            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                # bbox = cv2.selectROI(frame, False)
        else:
            if tracker_init:
                tracker.clear()
                del tracker
                tracker_init = False
            cv2.putText(frame, "No Detection", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0),2)
        if ok==True:
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        count += 1

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
