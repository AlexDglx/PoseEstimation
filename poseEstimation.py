import cv2
import time
import PoseModule as pm
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import pandas as pd
from vidgear.gears import WriteGear
from cameraCalibration import cameraDetails, jsonConfigFile

DATAPOINTS_REF = "reference_points.csv"
DATAPOINTS_ATTEMPT = "attempt_points.csv"
ACCURACY_DATA_SET = "accuracy.csv"
ACCURACY_DATA_HEADER= ["Accuracy %"]
DANCE_ANALYSIS = 'accuracy_analysis.mp4'

HEADER = ['nose','left_eye_inner','left_eye','left_eye_outer',
        'right_eye_inner','right_eye','right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder','right_shoulder','left_elbow','right_elbow',
        'left_wrist', 'right_wrist','left_pinky','right_pinky',
        'left_index','right_index','left_thumb','right_thumb','left_hip','right_hip',
        'left_knee','right_knee','left_ankle','right_ankle','left_heel','right_heel',
        'left_foot_index', 'right_foot_index'
        ]

detector = pm.PoseDetector()
data_reference = []
data_attempt = []
data_accuracy = []

# get camera details


def main():

    path_master = 'Media Pose Estimation/Dance BTS - Good.mp4'
    path_attempt = 'Media Pose Estimation/Danse BTS - Bad.mp4'
    cap1 = cv2.VideoCapture(path_master)
    cap2 = cv2.VideoCapture(path_attempt)
    pTime = 0

    #create video
    frame_rate = int(cap1.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    output_params =  {"-vcodec":"libx264", "-crf": 0, "-preset": "fast", "-r":f'{frame_rate}',"-pix_fmt":"yuv420p", "-vf":f'setpts= {0.78}*PTS'}
    writer = WriteGear(output=DANCE_ANALYSIS, compression_mode=True, logging=True, **output_params)

    if not cap1.isOpened() and not cap2.isOpened():
        print("Error: Could not open video.")
    # read camera details

    else:
        while True:
            ret1, img1 = cap1.read()
            ret2, img2 = cap2.read()



            if not ret1 or not ret2:
                print("End of videos")
                break
            original_img1, oiginal_img2 = img1.copy(), img2.copy()

            cv2.putText(original_img1, f'Reference', (150,100), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            cv2.putText(oiginal_img2, f'Attempt', (150,100), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            original_footage = np.hstack((original_img1, oiginal_img2 ))

            img1 = detector.findPose(img1)
            lmList1=detector.getPosition(img1, color=(0,255,0))
            data_reference.append(lmList1)

            img2 = detector.findPose(img2)
            lmList2 = detector.getPosition(img2, color=(0,0,255))
            data_attempt.append(lmList2)

            cTime = time.time()
            fps = 1 / (cTime - pTime) 
            pTime = cTime
            cv2.putText(img1, f'{str(int(fps))} fps', (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            positions =  33
            accurate_position = 0

            cv2.putText(img1, f'Dance Accuracy: ', (15,550), cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,0),1)
            # average position for each point
            for l1, l2 in zip(lmList1, lmList2):
                # cx cy cz visibility
                mtx = [int((l1[0]+l2[0])/2), int((l1[1]+l2[1])/2), int((l1[2]+l2[2])/2), int((l1[3]+l2[3])/2)]
                
                cv2.circle(img1, (mtx[0], mtx[1]),4, (67,135,226), cv2.FILLED)
                if l2[0]/l1[0] < 1 :
                    cv2.putText(img2, f'{round(100*l2[0]/l1[0],2)} %', (l1[0]+15,l2[1]), cv2.FONT_HERSHEY_SIMPLEX,0.25,(255,255,255), 1)
                else: 
                    cv2.putText(img2, f'{round(100*l1[0]/l2[0],2)} %', (l1[0]+15,l2[1]), cv2.FONT_HERSHEY_SIMPLEX,0.25,(0,0,255),1)
                   
                if distance((l1[0],l2[0]),(l1[1],l2[1])) <= 25:
                    cv2.line(img1, (l1[0],l1[1]), (l2[0],l2[1]), (255,0,0),1,cv2.LINE_AA)
                    accurate_position+=1
                else:
                    cv2.line(img1, (l1[0],l1[1]), (l2[0],l2[1]), (0,0,255),2,cv2.LINE_AA)
            
            q = round(100*accurate_position/positions,2)
            data_accuracy.append([q])

            if q > 80 :
                cv2.putText(img1, f'{q} %', (110,550), cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,255,0),1)
            else:
                cv2.putText(img1, f'{q} %', (110,550), cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,0,255),1)
            legend_left = f'-red line: distance error > 25\n-blue line : distance error <= 25pts\n-green landmarks: reference dance points > 25pts\n-red landmarks: dance attempt\n-orange landmarks: mean distance green & red landmarks)'
            legend_right = f'-White (%): pose landmark within(**)\n-Red (%): pose landmark out of range\n(**) Work in progress measurement.'           
            y0, dy = 570,10
            for i, line in enumerate(legend_left.split('\n')):
                y = y0 + i*dy
                cv2.putText(img1,line, (15,y), cv2.FONT_HERSHEY_SIMPLEX,0.30,(0,0,0),1)

            for i, line in enumerate(legend_right.split('\n')):
                y = y0 + i*dy
                cv2.putText(img2,line, (15,y), cv2.FONT_HERSHEY_SIMPLEX,0.30,(0,0,0),1)            
            
            cv2.putText(img1, f'Pose Reference', (150,100), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            cv2.putText(img2, f'Pose Attempt', (150,100), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            img_hz = np.hstack((img1, img2))
            img_final = np.vstack((original_footage,img_hz)) 

            writer.write(img_final)

            cv2.imshow("Comparison good/bad", img_final)

            if cv2.waitKey(25) & 0xFF == ord('q') :
                break

        cap1.release()
        cap2.release()
        writer.close()
        cv2.destroyAllWindows()

        with open(DATAPOINTS_REF, mode="w", newline="") as f :
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(data_reference)
        
        with open(DATAPOINTS_ATTEMPT, mode="w", newline="") as f :
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(data_attempt)

        with open(ACCURACY_DATA_SET, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ACCURACY_DATA_HEADER)
            writer.writerows(data_accuracy)

def distance(x:tuple, y:tuple):
    d = math.sqrt(math.pow(((x[1]-x[0])), 2) + math.pow(((y[1]-y[0])), 2))
    return d


if __name__ == "__main__":
    #main()
    #show accuracy graph
    img_path = "screenshot-reference.png"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    w, h= img.shape[:2]
    #read json calibration data 
    jsonPhoneCalibPath = "Apple_iPhone 12 Pro MAX_Ultra Wide__4k_16by9_3840x2160-30.00fps.json"
    iphone_calib_data = jsonConfigFile.readJson(jsonPhoneCalibPath)
    cameraMtx = np.array(iphone_calib_data["fisheye_params"]["camera_matrix"])
    distortion_coef = np.array(iphone_calib_data["fisheye_params"]["distortion_coeffs"])

    # CAMERA MATRIX WORKS FOR A 3840 2160 -> 640 - 360
    # Let's upscale the image
    up_width, up_height = iphone_calib_data["calib_dimension"]["w"],iphone_calib_data["calib_dimension"]["h"]
    up_points = (up_height,up_width)
    resized_up = cv2.resize(img, up_points, interpolation=cv2.INTER_LINEAR)
    newcameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMtx, distortion_coef, up_points,1,up_points)
    
    mapx, mapy = cv2.initUndistortRectifyMap(newcameraMatrix,distortion_coef,None,cameraMtx, up_points,5)
    dst_remap = cv2.remap(resized_up, mapx, mapy, cv2.INTER_LINEAR)

    dst = cv2.undistort(resized_up, cameraMtx, distortion_coef, None, newcameraMatrix)

    img_hz = np.hstack((resized_up, dst,dst_remap))
    cv2.imshow("Original / Undistort", img_hz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    




    # image, calc newcameramatrix and roi   
    


    '''
    data = pd.read_csv(ACCURACY_DATA_SET)
    values = data['Accuracy %']
    plt.figure(figsize=(10,6))
    plt.bar(range(len(values)), values, color='skyblue')
    plt.ylabel('Accuracy %')
    plt.xlabel('Frame Index')
    plt.title('Dancing Accuracy')
    plt.savefig('Accuracy Chart.png')
    plt.show()
    '''





