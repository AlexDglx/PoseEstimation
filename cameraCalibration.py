# Alex Degallaix - 2023
# All rights reserved
# Purpose : This script is to calibrate a stereoscopic vision system. With camera0 => Left camera and camera1 => Right camera
# Inspired by a stereo vision system calibration code adapted to Volumetric Video project :  https://github.com/TemugeB/python_stereo_camera_calibrate/blob/main/calib.py

import cv2 
import numpy as np
import glob
from tqdm import tqdm
from datetime import datetime
import pickle
import json
import time

class jsonConfigFile:
    #read json file
    def readJson(jsonPath: str) -> dict:
        """Read JSON camera configuration file.

        param p1: describe about parameter p1

        jsonPath: str
            Document string 
        """
        with open(jsonPath, "r") as file:
            a = json.load(file)
            file.close()
        return a
    #write date and time - for last modification
    def writeJson(jsonPath:str,newData:dict) -> any:
        with open(jsonPath,"w") as newJson:
            json.dump(newData,newJson,indent=4)
            newJson.close()

        a = jsonConfigFile.readJson(jsonPath)
        a['file_timestamp']['date_file_last_update']=  datetime.now().strftime("%Y-%m-%d")
        a['file_timestamp']['time_file_last_update'] = datetime.now().strftime("%H:%M:%S")
        a['file_timestamp']['timestamp'] = f'{a["file_timestamp"]["date_file_last_update"]}  {a["file_timestamp"]["time_file_last_update"]}'

        with open(jsonPath,"w") as newJson:
            json.dump(a,newJson,indent=4)
            newJson.close()
        return newJson

class stereoscopicImageCapture:
    def twoCameraCapture(camera_config:dict, json_configuration_path:str) -> any:
        try:
            print("Reading JSON file...")
            if camera_config is not None:

                desc = camera_config["camera_settings"]["id"]["sensor_sn"],camera_config["camera_settings"]["id"]["description"]
                print(f'Camera details: Sensor SN -> {desc[0]} Sensor Model (Mfg) -> {desc[1]}')

                root = camera_config["camera_settings"]["calibration"]["mono"]
                left_picture_data = root["camera0"]["calibration_folder_path"]
                right_picture_data = root["camera1"]["calibration_folder_path"]
                leftCamera = cv2.VideoCapture(camera_config["camera_settings"]["camera_properties"]["camera0"])
                rightCamera = cv2.VideoCapture(camera_config["camera_settings"]["camera_properties"]["camera1"])
                
                #counter for 
                img_counter = 0
                while True:
                    ret1, leftFrame = leftCamera.read()
                    ret2, rightFrame = rightCamera.read()
                    if not ret1 or not ret2:
                        print('Failed to grab both frames')
                        break
                    numpy_horizontal = np.hstack((leftFrame, rightFrame))
                    cv2.imshow("Left & Right Cameras", numpy_horizontal)
                    k = cv2.waitKey(1)
                    if k%256 ==27:
                        print('Escape hit, closing image capture...')
                        break
                    elif k%256 == 32:
                        if (img_counter <= camera_config["camera_settings"]["calibration"]["frames"]["stereo_calibration_frames"]):
                            left_img =f'{left_picture_data}'"/leftCamera{}.png".format(img_counter)
                            right_img = f'{right_picture_data}'"/rightCamera{}.png".format(img_counter)
                            cv2.imwrite(left_img, leftFrame)
                            cv2.imwrite(right_img, rightFrame)
                            print("{} written".format(left_img),"{} written".format(right_img))
                            img_counter += 1
                        else:
                            print('Reached maximum number of calibration frames.')
                            
                leftCamera.release()
                rightCamera.release()
                try:
                    #date
                    camera_config['camera_settings']['calibration']['mono']['camera0']['date']=  datetime.now().strftime("%Y-%m-%d")
                    camera_config['camera_settings']['calibration']['mono']['camera1']['date'] = datetime.now().strftime("%Y-%m-%d")
                    #time
                    camera_config['camera_settings']['calibration']['mono']['camera0']['time'] = datetime.now().strftime("%H:%M:%S")
                    camera_config['camera_settings']['calibration']['mono']['camera1']['time'] = datetime.now().strftime("%H:%M:%S")
                    
                    camera_config['file_timestamp']['last_task'] = f'Calibration image capture'
                    print(camera_config)

                    jsonConfigFile.writeJson(json_configuration_path,camera_config)
                    return print('Stereo calibration image capture done.')
                except FileExistsError:
                    print('File does not exist.')
                cv2.destroyAllWindows()
            else :
                print("Check camera JSON file.")
        except FileNotFoundError:
            print("File does not exist, check path.")

    def verification_image_capture(camera_config:dict, json_configuration_path:str) -> any:
        try:
            print("Reading JSON file...")
            if camera_config is not None:
                desc = camera_config["camera_settings"]["id"]["sensor_sn"],camera_config["camera_settings"]["id"]["description"]
                print(desc)
                root = camera_config["camera_settings"]["calibration"]["mono"]
                left_picture_data = root["camera0"]["verification_folder_path"]
                right_picture_data = root["camera1"]["verification_folder_path"]
                leftCamera = cv2.VideoCapture(camera_config["camera_settings"]["camera_properties"]["camera0"])
                rightCamera = cv2.VideoCapture(camera_config["camera_settings"]["camera_properties"]["camera1"])
                
                img_counter = 0
                while True:
                    ret1, leftFrame = leftCamera.read()
                    ret2, rightFrame = rightCamera.read()
                    if not ret1 or not ret2:
                        print('Failed to grab both frames')
                        break
                    numpy_horizontal = np.hstack((leftFrame, rightFrame))
                    cv2.imshow("Left & Right Cameras", numpy_horizontal)
                    k = cv2.waitKey(1)
                    if k%256 ==27:
                        print('Escape hit, closing image capture...')
                        break
                    elif k%256 == 32:
                        if (img_counter <= camera_config["camera_settings"]["calibration"]["frames"]["verification_frame"]):
                            left_img =f'{left_picture_data}'"/leftCamera_verficiation{}.png".format(img_counter)
                            right_img = f'{right_picture_data}'"/rightCamera_verification{}.png".format(img_counter)
                            cv2.imwrite(left_img, leftFrame)
                            cv2.imwrite(right_img, rightFrame)
                            print("{} written".format(left_img),"{} written".format(right_img))
                            img_counter += 1
                            return (left_img,right_img)
                        else:
                            print('Reached maximum number of verification frames.')
                leftCamera.release()
                rightCamera.release()
                
                try:
                    #date
                    camera_config['camera_settings']['calibration']['mono']['camera0']['date']=  datetime.now().strftime("%Y-%m-%d")
                    camera_config['camera_settings']['calibration']['mono']['camera1']['date'] = datetime.now().strftime("%Y-%m-%d")
                    #time
                    camera_config['camera_settings']['calibration']['mono']['camera0']['time'] = datetime.now().strftime("%H:%M:%S")
                    camera_config['camera_settings']['calibration']['mono']['camera1']['time'] = datetime.now().strftime("%H:%M:%S")
                    
                    camera_config['file_timestamp']['last_task'] = 'Verification image capture'
                    jsonConfigFile.writeJson(json_configuration_path,camera_config)

                except FileExistsError:
                    print('File does not exist.')
                cv2.destroyAllWindows()
                
            else :
                print("Check camera JSON file.")
        except FileNotFoundError:
            print("File does not exist, check path.")

            
class stereoscopicCalibration:
    #single camera calibration function

    def checkerboardCalibration(camera_config:dict, json_configuration_path:str) -> tuple:
        try:
            #getting camera details and checkerboard details
            if camera_config is not None:
                print("Reading JSON file...")
                root = camera_config["camera_settings"]["checkerboard"]
                checkerDimension = root["checkersize"]
                nbCol =  root["checker_dimensions"][1]
                nbRows =  root["checker_dimensions"][0]

                CHECKERBOARD = (nbCol-1, nbRows-1) 
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001) 
                # Vector for 3D points array
                threedpoints = [] 
                # Vector for 2D points array
                twodpoints = [] 
                
                #  3D points real world coordinates 
                objectp3d = np.zeros((1, CHECKERBOARD[0]  
                        * CHECKERBOARD[1],  
                        3), np.float32) 
                objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
                                0:CHECKERBOARD[1]].T.reshape(-1, 2)*checkerDimension

                try:
                    root= camera_config['camera_settings']['calibration']['mono']
                    available_cameras = [root['camera0']['camera_name'], root['camera1']['camera_name']]
                    camera_name = input(f'Input camera Name:\n(-> Note: Available camera names: {available_cameras[0]}, {available_cameras[1]})\n')
                    path = camera_config['camera_settings']['calibration']['mono'][camera_name]['calibration_folder_path']
                    images = sorted(glob.glob(f'{path}/*png'))
                    
                    if len(images) != 0:
                        x= 0
                        y= 0
                        h= camera_config["camera_settings"]["camera_properties"]["frame_dimensions"][0]
                        w= camera_config["camera_settings"]["camera_properties"]["frame_dimensions"][1]
                        processed_images = 0
                        for filename in tqdm(images, desc="Processing Images"): 
                            image = cv2.imread(filename)[y:y+h,x:x+w]
                            grayColor =  cv2.cvtColor(cv2.GaussianBlur(image, (5,5),0), cv2.COLOR_BGR2GRAY)
                            ret, corners = cv2.findChessboardCorners( grayColor, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE) 
                            if ret == True: 
                                threedpoints.append(objectp3d) 
                                corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria) 
                                twodpoints.append(corners2) 
                                image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
                                processed_images+=1
                            print(f'successfully processed images : {processed_images}')       
                        ret, matrix, distortion, r_vecs, t_vecs= cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)
                        print(f'Camera Matrix:\n{matrix}\nDistortion coefficients:\n{distortion}\nRotational Vectors:\n{r_vecs}\nTranslation Vectors:\n{t_vecs}')
                        #refined camera matrix
                        h, w = image.shape[:2]
                        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion,(w,h),1,(w,h))
                        #from camera matrix undistort image
                        dst = cv2.undistort(image, matrix, distortion, None, newcameramatrix)
                        #crop the image to restricted region
                        x, y, w, h = roi
                        mapx, mapy = cv2.initUndistortRectifyMap(newcameramatrix,distortion,None,matrix, (w,h),5)
                        dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                        #re-projection error
                        mean_error = 0
                        for i in range(len(objectp3d)):
                            imgpoints2d, _ = cv2.projectPoints(objectp3d[i],r_vecs[i],t_vecs[i],matrix,distortion)
                            error =cv2.norm(twodpoints[i], imgpoints2d,cv2.NORM_L2)/len(imgpoints2d)
                            mean_error += error
                        cal_error=mean_error/len(objectp3d)
                        #refined distortion matrix
                        cameraDataTable = {
                                        "RawCameraMatrix": matrix, 
                                        "DistortionMatrix":distortion, 
                                        "Rotation_vectors":r_vecs,
                                        "Translation_vectors":t_vecs,
                                        "NewCameraMatrix":newcameramatrix,
                                        "TunedDistX": mapx,
                                        "TunedDistY": mapy,
                                        "CalibrationError":cal_error
                                        }
                        #storing data in pickle file with date and name
                        print(cameraDataTable)

                        calibration_path = cameraDetails.saveCalibrationFile(cameraDataTable,input("File name : "))

                        print(f'Calibration Error: {round(cameraDataTable["CalibrationError"],5)} \nDetection success: {round(100*processed_images/len(images),3)}%')

                        try:
                            #date
                            camera_config['camera_settings']['calibration']['mono']['camera0']['date']=  datetime.now().strftime("%Y-%m-%d")
                            camera_config['camera_settings']['calibration']['mono']['camera1']['date'] = datetime.now().strftime("%Y-%m-%d")
                            #time
                            camera_config['camera_settings']['calibration']['mono']['camera0']['time'] = datetime.now().strftime("%H:%M:%S")
                            camera_config['camera_settings']['calibration']['mono']['camera1']['time'] = datetime.now().strftime("%H:%M:%S")
                            # calibration path 
                            camera_config['camera_settings']['calibration']['mono'][camera_name]['cal_filepath'] = calibration_path

                            camera_config['file_timestamp']['last_task'] = f'{camera_name} checkerboard calibration'

                            jsonConfigFile.writeJson(json_configuration_path,camera_config)

                            return (ret, matrix, distortion, r_vecs, t_vecs)

                        except FileNotFoundError:
                            print('Check sensor JSON file.')
                        return cameraDataTable
                    else :
                        print("Empty Folder")
                except FileNotFoundError:
                    print('Check file path.')
            else:
                print("File is empty. Check file.")
        except FileNotFoundError:
            print("file not found")

    def stereoCheckerboardCalibrate(camera_config:dict, json_configuration_path:str) -> np.array:
        """Calibrate the stereoscopic camera setup.

        method looks for checkerboard photo matching pairs.
        """
        try:
            if camera_config is not None:
                checkerDimension = camera_config["camera_settings"]["checkerboard"]["checkersize"]
                nbCol =  camera_config["camera_settings"]["checkerboard"]["checker_dimensions"][1]
                nbRow =  camera_config["camera_settings"]["checkerboard"]["checker_dimensions"][0]                
                root = camera_config["camera_settings"]["calibration"]["mono"]
                LeftCameraFolderPath = root["camera0"]["calibration_folder_path"]
                RightCameraFolderPath = root["camera1"]["calibration_folder_path"]
                leftCalibrationPath = root['camera0']["cal_filepath"]
                rightCalibrationPath = root['camera1']["cal_filepath"]
                
                camDetails_left = cameraDetails.readCalibrationFile(leftCalibrationPath)
                camDetails_right = cameraDetails.readCalibrationFile(rightCalibrationPath)

                CHECKERBOARD = (nbCol-1, nbRow-1) 
                #read synced frames
                left_img = sorted(glob.glob(f'{LeftCameraFolderPath}/*.png'))
                right_img = sorted(glob.glob(f'{RightCameraFolderPath}/*.png'))
                #read frames
                left_cam  = [cv2.imread(img,1) for img in tqdm(left_img, desc="Reading left camera images")]
                right_cam = [cv2.imread(img, 1) for img in tqdm(right_img, desc="Reading right camera images")]
                #place to fine tune stereo image calibration
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
                #  3D points real world coordinates 
                objp = np.zeros((1, CHECKERBOARD[0]  
                        * CHECKERBOARD[1],  
                        3), np.float32) 
                objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
                                0:CHECKERBOARD[1]].T.reshape(-1, 2)*checkerDimension

                width = left_cam[0].shape[1]
                height = left_cam[0].shape[0]
                # Vector coordinates for 2D points array
                imagePoints_left = [] 
                imagePoints_right= []
                # Vector coordinates for 2D points array
                objectPoints = []
                checkerboard_found = 0

                for leftFrame, rightFrame in zip(left_cam,right_cam):
                    left_gray = cv2.cvtColor(leftFrame,cv2.COLOR_BGR2GRAY)
                    right_gray = cv2.cvtColor(rightFrame,cv2.COLOR_BGR2GRAY)
                    c_ret1, corners1 = cv2.findChessboardCorners(left_gray,(CHECKERBOARD[1],CHECKERBOARD[0]), None)
                    c_ret2, corners2 = cv2.findChessboardCorners(right_gray,(CHECKERBOARD[1],CHECKERBOARD[0]), None)

                    if c_ret1 and c_ret2:
                        corners1 = cv2.cornerSubPix(left_gray,corners1, (11, 11), (-1, -1), criteria)
                        corners2 = cv2.cornerSubPix(right_gray,corners2, (11, 11), (-1, -1), criteria)

                        cv2.drawChessboardCorners(leftFrame,(CHECKERBOARD[1],CHECKERBOARD[0]),corners1, c_ret1)
                        cv2.drawChessboardCorners(rightFrame,(CHECKERBOARD[1],CHECKERBOARD[0]),corners2, c_ret2)

                        objectPoints.append(objp)
                        imagePoints_left.append(corners1)
                        imagePoints_right.append(corners2)
                        checkerboard_found +=1
                        horizontal = np.hstack((leftFrame, rightFrame))
                        cv2.imshow('Checkerboard Find Left/Right Sync',horizontal)
                        cv2.waitKey(0)
                        cv2.destroyWindow('Checkerboard Find Left/Right Sync')
                #print(objectPoints)
                #flag |= cv2.CALIB_FIX_INTRINSIC CALIB_USE_INTRINSIC_GUESS
                flag = 0
                flag |= cv2.CALIB_FIX_INTRINSIC
                stereocalibration_flags  = cv2.CALIB_USE_INTRINSIC_GUESS
                ret, CM1, dist_left, CM2, dist_right, R, T, E, F =  cv2.stereoCalibrate(objectPoints, 
                                                                                    imagePoints_left, 
                                                                                    imagePoints_right,
                                                                                    camDetails_left["RawCameraMatrix"],
                                                                                    camDetails_left["RawCameraMatrix"],
                                                                                    camDetails_right["NewCameraMatrix"],
                                                                                    camDetails_right["DistortionMatrix"],
                                                                                    (width, height),
                                                                                    criteria = criteria,
                                                                                    flags = stereocalibration_flags)
                #CALIB_ZERO_DISPARITIES
                R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(CM1, dist_left, CM2, dist_right, (width, height), R, T, flags=cv2.CALIB_CB_ADAPTIVE_THRESH, alpha=0.9)
                #The Q matrix of the horizontal stereo setup. 
                #This matrix is used for reprojecting a disparity tensor to the corresponding point cloud.
                #Note that this is in a general form that allows different focal lengths in the x and y direction.
                stereoDetails = {
                                "RMS": ret,
                                "StereoRotationVec": R, 
                                "StereoTranslationVec":T,
                                "StereoEssentialMtx": E,
                                "StereoFundamentalMtx": F,
                                "LeftCameraMtx":CM1,
                                "LeftCameraDist":dist_left,
                                "RightCameraMtx":CM2,
                                "RightCameraDist":dist_right,
                                "Left_Rectified_Rot_vecs": R1,
                                "Right_Rectified_Rot_vecs": R2,
                                "Left_Rectified_Translation_vecs": P1,
                                "Right_Rectified_Translation_vecs":P2,
                                "Q_Projection_Matrix":Q
                                }
                stereo_calibration_path = cameraDetails.saveCalibrationFile(stereoDetails,input("Stereo Cal File name: "))
                print(f'Chekerboard found success rate: \n{100*(checkerboard_found/len(left_cam))}%', checkerboard_found,len(left_cam),E, ret)

                try:
                    #date
                    camera_config['camera_settings']['calibration']['stereo']['date']=  datetime.now().strftime("%Y-%m-%d")
                    #time
                    camera_config['camera_settings']['calibration']['stereo']['time'] = datetime.now().strftime("%H:%M:%S")
                    # stereo calibration path 
                    camera_config['camera_settings']['calibration']['stereo']['cal_filepath'] = stereo_calibration_path

                    camera_config['file_timestamp']['last_task'] = 'Checkerboard Stereo calibration'

                    jsonConfigFile.writeJson(json_configuration_path,camera_config)

                except FileNotFoundError:
                    print('Check sensor JSON file.')
                
                return R, T
            else:
                print("File is empty. Check file.")
                
        except FileNotFoundError:
            print("File not found")
    
    def homogeonous_matrix(R,t):
        """Get homogeouns matrix from rotation vectors and translation vectors
        
        """
        P = np.zeros((4,4))
        P[:3,:3] = R
        P[:3, 3] = t.reshape(3)
        P[3,3] = 1
        return P
        
    def projectionMatrix(mtx, R,T) -> any:
        """
            Get projection matrix from homogeonus matrix
        """
        P = mtx@stereoscopicCalibration.homogeonous_matrix(R,T)[:3,:]
        return P
        
class checkCalibration:
    def cal_verif(camera_config:dict, json_configuration_path: str, R_W0:np.ndarray, T_W0:np.ndarray, R_W1:np.ndarray, T_W1:np.ndarray, zshift=50.) -> np.ndarray:
        """
            Verify stereo camera calibration 
        """
        try:
            if camera_config is not None:
                #get stream values
                stream = camera_config['camera_settings']['camera_properties']
                camera_stream = [stream['camera0'],stream['camera1']]
                
                #get calibration file paths camera matrix and distortion matrix for camera0 and camera1
                camera_mono_root = camera_config['camera_settings']['calibration']['mono']
                camera0_data = cameraDetails.readCalibrationFile(camera_mono_root['camera0']['cal_filepath'])
                camera1_data = cameraDetails.readCalibrationFile(camera_mono_root['camera1']['cal_filepath'])
                [cmtx0, dist0]= [camera0_data['NewCameraMatrix'],camera0_data['DistortionMatrix']]
                [cmtx1, dist1]= [camera1_data['NewCameraMatrix'],camera1_data['DistortionMatrix']]

                #read extrinsic calibration file for values R0,T0, (R1, T1) values from stereo rotation and translation
                
                #get projected matrix for each camera
                P0 = stereoscopicCalibration.projectionMatrix(cmtx0,R_W0,T_W0)
                P1 = stereoscopicCalibration.projectionMatrix(cmtx1,R_W1,T_W1)
                #Coordinate axes in 3D space
                coordinates = np.array([
                                        [0.,0.,0.],
                                        [1.,0.,0.],
                                        [0.,1.,0.],
                                        [0.,0.,1.]
                                        ])
                zshift  = np.array([0.,0.,zshift]).reshape((1,3))
                #increase the coordinates axes and shift in z direction
                draw_axes_points  = 60*coordinates + zshift
                #project 3D points to each camera views manually. or with cv2.projectPoints()
                # Homogenous coordinates
                pixel_points_left_cam = []
                pixel_points_right_cam = []
                for p in tqdm(draw_axes_points, desc="Vector coordinates..."):
                    X = np.array([p[0],p[1],p[2],1.])

                    #projection to left camera or camera 0
                    uv = P0 @ X
                    uv = np.array([uv[0], uv[1]])/uv[2]
                    pixel_points_left_cam.append(uv)
                    #projection to right camera or camera 1
                    uv = P1 @ X
                    uv = np.array([uv[0], uv[1]])/uv[2]
                    pixel_points_right_cam.append(uv)
                pixel_points_left_cam = np.array(pixel_points_left_cam)
                pixel_points_right_cam = np.array(pixel_points_right_cam)

                #start videostreams
                camera0= cv2.VideoCapture(camera_stream[0])
                camera1= cv2.VideoCapture(camera_stream[1])

                while True:
                    ret0, left_frame = camera0.read()
                    ret1, right_frame = camera1.read()
                    if not ret0 or not ret1:
                        print('Videostream is unavailable')
                        quit()
                    #rbg colours for axis
                    colours = [(0,0,255),(0,255,0),(255,0,0)]
                    #draw projection coordinates to camera 0 (left camera)
                    origin = tuple(pixel_points_left_cam[0].astype(np.int32))
                    for col, _p in zip(colours, pixel_points_left_cam[1:]):
                        _p = tuple(_p.astype(np.int32))
                        cv2.line(left_frame,origin,_p,col,2)
                    #draw projection coordinates to camera 1 (right camera)
                    origin = tuple(pixel_points_right_cam[0].astype(np.int32))
                    for col, _p in zip(colours, pixel_points_right_cam[1:]):
                        _p = tuple(_p.astype(np.int32))
                        cv2.line(right_frame,origin,_p,col,2)
                    hz_frames= np.hstack((left_frame, right_frame))
                    cv2.imshow('World Coordinates', hz_frames)
                    #Save world reference frames
                    root = camera_config['camera_settings']['calibration']['mono']
                    world_frame_projection_path = [root['camera0']['world_projection'],root['camera1']['world_projection']]        
                    cv2.imwrite(f'{world_frame_projection_path[0]}'"/Cam0_world_projection.png", left_frame)
                    cv2.imwrite(f'{world_frame_projection_path[1]}'"/Cam1_world_projection.png", right_frame)
                    k = cv2.waitKey(1)
                    print('Stereo camera calibration and verification completed.')
                    
                    if k==27:
                        break

                    try:                        #date
                        ###### ADD PROJECTION FRAMES IN FILE
                        camera_config['camera_settings']['calibration']['stereo']['date']=  datetime.now().strftime("%Y-%m-%d")
                        #time
                        camera_config['camera_settings']['calibration']['stereo']['time'] = datetime.now().strftime("%H:%M:%S")
                        # stereo calibration path 
                        camera_config['file_timestamp']['last_task'] = 'Stereo calibration/verification and world projection'

                        jsonConfigFile.writeJson(json_configuration_path,camera_config)

                    except FileNotFoundError:
                            print('Check sensor JSON file.')
                    return P0, P1
                camera0.release()
                camera1.release()
                cv2.destroyAllWindows()
            else:
                print("Empty file. Check file")
        except FileNotFoundError:
            print('File not found')
    
    def get_world_origin(camera_config:dict) -> np.ndarray:
            """
            Get world origin for camera0 from extrinsic values
            """
            try:
                frame = cv2.imread(input('Path to left reference camera frame \n-> (note: frame to use as calibration for world origin reference):\n'),1)
                try:
                #getting camera details and checkerboard details
                    if camera_config is not None:
                        print("Reading JSON file...")
                        cal_file_path = camera_config['camera_settings']["calibration"]["mono"]['camera0']["cal_filepath"]
                        calibration_data = cameraDetails.readCalibrationFile(cal_file_path)
                        [camera_mtx, dist] =  [calibration_data['NewCameraMatrix'], calibration_data['DistortionMatrix']]
                        root = camera_config["camera_settings"]["checkerboard"]
                        #world scaling = checker size
                        checker_data = [root["checker_dimensions"][1],root["checker_dimensions"][0],root["checkersize"]]
                        print(checker_data)
                        CHECKERBOARD = (checker_data[0]-1, checker_data[1]-1) 
                        objectp3d = np.zeros((1, CHECKERBOARD[0]  
                        * CHECKERBOARD[1],  
                        3), np.float32) 
                        objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
                                0:CHECKERBOARD[1]].T.reshape(-1, 2)*checker_data[2]

                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        try:
                            ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD[1],CHECKERBOARD[0]),None)
                            cv2.drawChessboardCorners(frame,(CHECKERBOARD[1],CHECKERBOARD[0]), corners,ret)
                            ret, rvec, tvec = cv2.solvePnP(objectp3d,corners,camera_mtx,dist)
                            R, _ = cv2.Rodrigues(rvec)
                            cv2.imshow('Calibration Reference Frame', frame)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            return R, tvec
                        except :
                            print('Image does not contain a checkerboard target')                      
                    else:
                        print('File is empty or missing values. Check config file.') 
                except FileNotFoundError:
                    print("Check sensor configuration file")
            except:
                print('Error with selected image. Select an image with a calibration grid.')
    
    #define the right camera extrinsic values as the world origin
    def get_camera1_world_transform(camera_data:dict, R_W0:np.ndarray,T_W0:np.ndarray, json_configuration_path:str) -> np.ndarray:
        """Using Rodrigues transform:
                -> Project camera0 coordinates as origin, then project coordinates to camera1. 
        """
        frames = stereoscopicImageCapture.verification_image_capture(camera_data, json_configuration_path)
        
        frame0 = cv2.imread(frames[0])
        frame1 = cv2.imread(frames[1])
        root_mono = camera_data['camera_settings']['calibration']['mono']
        camera0_data = cameraDetails.readCalibrationFile(root_mono['camera0']['cal_filepath'])
        camera1_data = cameraDetails.readCalibrationFile(root_mono['camera1']['cal_filepath'])
        [cmtx0, dist0] = [camera0_data['NewCameraMatrix'],camera0_data['DistortionMatrix']]
        [cmtx1, dist1] = [camera1_data['NewCameraMatrix'],camera1_data['DistortionMatrix']]
        #Stereo file

        root_stereo = camera_data['camera_settings']['calibration']['stereo']
        camera_stereo = cameraDetails.readCalibrationFile(root_stereo['extrinsic_filepath'])
        print(f'Data For:\n{camera_stereo}')
        [R_01,T_01] = [camera_stereo['stereoRot'],camera_stereo['stereoTr']]

        unitv_points = 50 * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
        colours = [(0,0,255),(0,255,0),(255,0,0)]
        #project origin points to frame 0
        points, _ = cv2.projectPoints(unitv_points,R_W0,T_W0,cmtx0,dist0)
        points = points.reshape((4,2)).astype(np.int32)
        origin = tuple(points[0])
        for col, _p in zip(colours, points[1:]):
            _p = tuple(_p.astype(np.int32))
            cv2.line(frame0,origin,_p,col,2)

        #project origin points to frame 1
        R_W1 = R_01 @ R_W0
        T_W1 = R_01 @ T_W0 + T_01
        points, _ = cv2.projectPoints(unitv_points, R_W1, T_W1, cmtx1,dist1)
        points = points.reshape((4,2)).astype(np.int32)
        origin = tuple(points[0])
        for col, _p in zip(colours, points[1:]):
            _p = tuple(_p.astype(np.int32))
            cv2.line(frame1,origin,_p,col,2)
        horizontal = np.hstack((frame0,frame1))
        
        #save world calibration data
        world_projection_path = checkCalibration.save_calibration_world(R_W0,T_W0,R_W1, T_W1)

        #save world projection frames (left and right)
        root = camera_data['camera_settings']['calibration']['mono']
        world_projection_frame_path = [root["camera0"]["world_projection"],root["camera1"]["world_projection"]]
        cv2.imwrite(f'{world_projection_frame_path[0]}/cam0_world_projection.png',frame0)
        cv2.imwrite(f'{world_projection_frame_path[1]}/cam1_world_projection.png',frame1)
        cv2.imshow('World Projection - Left & Right frames', horizontal)

        try:
        #date
            camera_data['camera_settings']['calibration']['stereo']['date']=  datetime.now().strftime("%Y-%m-%d")
            #time
            camera_data['camera_settings']['calibration']['stereo']['time'] = datetime.now().strftime("%H:%M:%S")
            # world calibration path 
            camera_data['camera_settings']['calibration']['stereo']['world_calibration_verification_filepath'] = world_projection_path
            camera_data['file_timestamp']['last_task'] = 'Stereo camera world projection'
            jsonConfigFile.writeJson(json_configuration_path,camera_data)
        except FileNotFoundError:
            print('Check sensor JSON file.')

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return R_W1, T_W1 
    
    #save intrinsic parameters
    def save_extrinsic_calibration_parameters(camera_config:dict, json_configuration_path: str) -> np.ndarray:
        """Save extrinsic camera parameters  World projection file
        """
        #camera0 for reference
            #np.eye() -> matrix identité
        R0, T0 = np.eye(3, dtype=np.float32), np.array([0.,0.,0.]).reshape((3,1))
        #get stereocal R and T vectors
        stereo_cal_path = camera_config["camera_settings"]['calibration']['stereo']['cal_filepath']
        stereo_data = cameraDetails.readCalibrationFile(stereo_cal_path)
        R1 , T1 = stereo_data['StereoRotationVec'],stereo_data['StereoTranslationVec'] 
        #save R, T, R0, T0 values to .dat - extrinsic calibration parameters
        extrinsic_data ={
            'camera0Rot': R0,
            'camera0Tr': T0,
            'stereoRot': R1,
            'stereoTr': T1,
            'extrinsic_cam0': np.c_[R0, T0],
            'extrinsic_cam1': np.c_[R1,T1]
            }
        stereo_extrinsic_calibration_path = cameraDetails.saveCalibrationFile(extrinsic_data, filename=input('Extrinsic data file name: '))

        try:
        #date
            camera_config['camera_settings']['calibration']['stereo']['date']=  datetime.now().strftime("%Y-%m-%d")
            #time
            camera_config['camera_settings']['calibration']['stereo']['time'] = datetime.now().strftime("%H:%M:%S")
            # world calibration path 
            camera_config['camera_settings']['calibration']['stereo']['extrinsic_filepath'] = stereo_extrinsic_calibration_path
            camera_config['file_timestamp']['last_task'] = 'Saved stereo camera extrinsic calibration values'
            
            jsonConfigFile.writeJson(json_configuration_path,camera_config)
       
        except FileNotFoundError:
            print('Check sensor JSON file.')

        return R0,T0, R1, T1
    
    def save_calibration_world(R_W0:np.ndarray, T_W0:np.ndarray, R_W1:np.ndarray, T_W1:np.ndarray) -> dict:
       """Save world projection R_W0, T_W0, R_W1 and T_W1 world projection file
       """
       world_rotation_translation ={
            'camera0_rotation_world': R_W0,
            'camera0_translation_world': T_W0,
            'camera1_rotation_world': R_W1,
            'camera1_translation_world': T_W1,
            'extrinsic_world_cam0': np.c_[R0, T0],
            'extrinsic_world_cam1': np.c_[R1,T1]
            }
       world_projection_path = cameraDetails.saveCalibrationFile(world_rotation_translation, filename=input('World projection calibration file name:\n'))
       return world_projection_path

class cameraDetails:
    def saveCalibrationFile(data:dict, filename:str) -> str:
        """Generic file dump and create string path to calibration file
            -> sensor_config.json is updated to the contian the new string path
        """
        calibration_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        str_current_datetime = str(calibration_date_time)
        name = f'Calibration/{str_current_datetime}_{filename}.dat'
        with open(name, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return str(name)

    def readCalibrationFile(PATH: str) -> list:
        with open(PATH, 'rb') as f:    
            list = pickle.load(f)
        return list
            
if __name__ == '__main__':
    start_timer = time.perf_counter()
    try:
        #read json file once
        json_file_main_path = f'{input("Camera details JSON file name: ")}.json'
        sensor_configuration_file= jsonConfigFile.readJson(json_file_main_path)
        
        #Step 1: Get calibration grid frames for calibration
        checkerboard_pictures_path_right = f'{sensor_configuration_file["camera_settings"]["calibration"]["mono"]["camera0"]["calibration_folder_path"]}/*.png'
        checkerboard_pictures_path_left = f'{sensor_configuration_file["camera_settings"]["calibration"]["mono"]["camera0"]["calibration_folder_path"]}/*.png'
        
        if (glob.glob(checkerboard_pictures_path_left) and
              glob.glob(checkerboard_pictures_path_right) is None):
            print(f'Writing in:\n-> {checkerboard_pictures_path_left}\n->{checkerboard_pictures_path_right}')
            stereoscopicImageCapture.twoCameraCapture(sensor_configuration_file,json_file_main_path)
        else:
            print(f'Checkerboard calibration pictures already stored.\n')
        #Step 2: Get individual camera intrinsic calibration values - Left Camera Calibration Camera Name == camera0
        print('Left camera intrinsic values calculation:')
        leftCamera_data = stereoscopicCalibration.checkerboardCalibration(sensor_configuration_file, json_file_main_path)
        print(f'Left Camera Calibration:\n{leftCamera_data}')
        
        #Step 2': Get individual camera intrinsic calibration values - Right Camera Calibration Camera Name == camera1
        print('Right camera intrinsic values calculation:')
        rightCamera_data = stereoscopicCalibration.checkerboardCalibration(sensor_configuration_file, json_file_main_path)
        print(f'Right Camera Calibration:\n{rightCamera_data}')

        #sleep 2sec to load left and right calibration .dat files
        #Step 3 : Get stereo camera calibration
        
        R, T = stereoscopicCalibration.stereoCheckerboardCalibrate(sensor_configuration_file,json_file_main_path)
        print(f'Stereo camera intrinsic values.\nStereo Rotation Vectors:\n{R}\nStereo Translation Vectors:\n{T}')
            
        #Step 4: Save calibration data where camera0=left camera and camera1=right camera. Left camera is the leading camera
        R0,T0, R1, T1 = checkCalibration.save_extrinsic_calibration_parameters(sensor_configuration_file,json_file_main_path)
        print(f'Extrinsic camera data:\nLeft camera Rotation and Translation vectors:\n{R0, T0}\Right camera Rotation and Translation vectors:\n{R1, T1}')

        R_W0, T_W0 = checkCalibration.get_world_origin(sensor_configuration_file)
        print(f'Rotation:\n{R_W0}\nTranslation:\n{T_W0}')
        R_W1, T_W1 = checkCalibration.get_camera1_world_transform(sensor_configuration_file,R_W0, T_W0, json_file_main_path)
        print(f'Rotation:\n{R_W1}\nTranslation:\n{T_W1}')
        
        #Camera calibration verification
        checkCalibration.cal_verif(sensor_configuration_file, json_file_main_path, R_W0, T_W0, R_W1, T_W1, zshift=0.)
    except FileExistsError:
        print('Check calibration file.')
    
    end_timer = time.perf_counter()

    execution_time = (end_timer-start_timer)

    print(f'Calibration execution time: {execution_time} seconds')
    
    
    
