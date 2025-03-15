import os
import re
import time
import cv2
import sys
import subprocess
import shutil
import numpy as np
import glob
import pickle
import sounddevice as sd

class TwinCams:

    def __init__(self):
        pass
    # *****************************************************************************************************************************
    @staticmethod
    def find_camera_ports(): # this funtion shows you ports that cameras are connected to.
        command=None
        if sys.platform.startswith("win"):
            command = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
        elif sys.platform.startswith("linux"):
            command = ["v4l2-ctl", "--list-devices"]
        else:
            print("this command is not supported for this os")
        if not command == None:
            subprocess.run(command,text=True, stderr=subprocess.STDOUT)
    # *****************************************************************************************************************************
    @staticmethod
    def cameras_orientation(camera_port_one,camera_port_two, stream =False): # for applications in stereo vision, it is 
                                                                             # important to know which camera is the left or right one
                                                                             # and this function returns (leftPort, rightPort).
        
        
        def pointsAVG(matched_points):                                      # this inner function is for some calculations
            avgx = sum(p[0] for p in matched_points) / len(matched_points)
            avgy = sum(p[1] for p in matched_points) / len(matched_points)
            return avgx, avgy
        
        cap1=cv2.VideoCapture(camera_port_one)
        cap2=cv2.VideoCapture(camera_port_two)
        
        while True:

            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()  
            key = cv2.waitKey(1) & 0xFF

            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(frame1,None)
            kp2, des2 = sift.detectAndCompute(frame2,None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1,des2,k=2)

            good = []
            matched_points_left = []
            matched_points_right = []

            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
                    # Get the keypoints coordinates
                    pt1 = kp1[m.queryIdx].pt  # (x, y) in left image
                    pt2 = kp2[m.trainIdx].pt  # (x, y) in right image
                    matched_points_left.append(pt1)
                    matched_points_right.append(pt2)

            avg_one=pointsAVG(matched_points_left)
            avg_two=pointsAVG(matched_points_right)
            
            if (avg_one[0]<avg_two[0]):
                camera_port_left=camera_port_one
                camera_port_right=camera_port_two
                text_one=f"Left(port: {camera_port_left})"
                text_two=f"Right(port: {camera_port_right})"
                
            else:
                camera_port_left=camera_port_two
                camera_port_right=camera_port_one
                text_one=f"Right(port: {camera_port_right})"
                text_two=f"Left(port: {camera_port_left})"

            print(f"port number {camera_port_left} camera is left and port number {camera_port_right} camera is right")

            print("*****************************************")
            if key == ord('q') or key == ord("Q") or stream == False:  # If 'q' key is pressed, it will exit
                print("Exiting function.")
                break
            

            height1, width1, channels1 = frame1.shape
            height2, width2, channels2 = frame2.shape

            font = cv2.FONT_HERSHEY_DUPLEX
            fontScale = 1
            color=(0,0,160)
            org1=(int(width1*1/3),int(height1/2))
            org2=(int(width2*1/3),int(height2/2))
            org_text=(int(width1/5),int(height1*4/5))
            cv2.putText(frame1,text_one,org1,font,fontScale,color)
            cv2.putText(frame2,text_two,org2,font,fontScale,color)
            cv2.putText(frame1,"press 'q' to exit",org_text,font,1.5,(0,255,10))
            cv2.putText(frame2,"press 'q' to exit",org_text,font,1.5,(0,255,10))
            cv2.imshow("First Cam", frame1)
            cv2.imshow("Second Cam", frame2)

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()       
        return (camera_port_left,camera_port_right)
    # *****************************************************************************************************************************
    @staticmethod
    def single_cam_take_photo(camera_port, relative_path = "Photos", replace = True, timer=0, max_number=0):# this function takes photos
                                                                                  # relatve path is the relative directory according 
                                                                                  # to the code direcotry that photos save there.
                                                                                  
                                                                                  # you can use "timer" and  "max_number" for auto-capturing without keyboard
                                                                                  # set "timer" and "max_number" equal to 0 if you dont need 
                                                                                  # auto-capturing or just leave them
        base_dir = os.path.dirname(os.path.abspath(__file__)) # handling the directory
        photo_directory = os.path.join(base_dir, relative_path, "SingleCam")
        photo_counter = 1 # to name photos

        if replace == False:  # handling directories
            if not os.path.exists(photo_directory):
                os.makedirs(photo_directory, exist_ok=True)
            else:
                existing_photos = [f for f in os.listdir(photo_directory) if re.match(r'photo(\d+)\.jpg', f)]
                if not existing_photos:
                    print("no photo has been already saved in the directory.")
                else:
                    numbers = [int(re.search(r'photo(\d+)\.jpg', f).group(1)) for f in existing_photos]
                    photo_counter = max(numbers) + 1
        else:
            if not os.path.exists(photo_directory):
                os.makedirs(photo_directory, exist_ok=True)
            else:
                shutil.rmtree(photo_directory)
                os.makedirs(photo_directory, exist_ok=True)
            

        cap=cv2.VideoCapture(camera_port) 
        print("Press 's' to capture a photo and 'q' to quit.")

        cap_time=time.time()
        old_time = time.time()
        while True:

            ret, org_frame = cap.read()
            frame = org_frame.copy()
            if not ret:
                print(f"failed to grab frame from camera or port.")
                break
            
            old_time, frame = TwinCams.fps_calc(old_time, frame)
            if timer!=0:
                font                   = cv2.FONT_HERSHEY_DUPLEX
                bottomLeftCornerOfText = (20,200)
                fontScale              = 1
                fontColor              = (20,20,255)
                thickness              = 1
                lineType               = 2
                cv2.putText(frame, f"timer: {round(timer - (time.time()-cap_time) , 2)} s", bottomLeftCornerOfText, font,\
                fontScale, fontColor, thickness, lineType)

            cv2.imshow(f"live cam: port{camera_port}", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in [ord('s'), ord('S')] or (timer - (time.time()-cap_time) < 0 and timer>0):  # take photo after pressing "s" or timer is activated
                filePath = os.path.join(photo_directory, f"photo{photo_counter}.jpg") 
                cv2.imwrite(filePath, org_frame)
                print(f"Photo saved in {filePath}.")
                photo_counter=photo_counter+1
                cap_time = time.time()
                if timer>0:
                    TwinCams.do_beep()

            if key in [ord('q'), ord('Q')] or (photo_counter>max_number and max_number>0):  # exits if "q" is pressed or the limit is reached
                print("Exiting function.")
                break

        cap.release()
        cv2.destroyAllWindows()
    # *****************************************************************************************************************************
    @staticmethod
    def stereo_cam_take_photo(camera_ports, relative_path= "Photos", replace = True, timer=0, max_number=0):# this function takes photos for stereo setup
                                                                                                  # relatve path is the relative directory according 
                                                                                                  # to the code direcotry that photos save there.
                                                                                                  
                                                                                                  # you can use "timer" and "max number" for auto capturing
                                                                                                  # if you set these variables 0, there is no auto capuring and
                                                                                                  # you must use keyboard to take photos
        camera_port_left, camera_port_right =  (camera_ports[0],camera_ports[1])
        base_dir = os.path.dirname(os.path.abspath(__file__))
        photo_directory_left = os.path.join(base_dir, relative_path, "Left")
        photo_directory_right = os.path.join(base_dir, relative_path, "Right")
        photo_counter = 1 # to name photos

        if replace == False:  # handling directories
            if not os.path.exists(photo_directory_left) or not os.path.exists(photo_directory_right):
                os.makedirs(photo_directory_left, exist_ok=True)
                os.makedirs(photo_directory_right, exist_ok=True)
            else:
                existing_photos_left = [f for f in os.listdir(photo_directory_left) if re.match(r'photo(\d+)\.jpg', f)]
                existing_photos_right = [f for f in os.listdir(photo_directory_right) if re.match(r'photo(\d+)\.jpg', f)]
                if not existing_photos_left or not existing_photos_right:
                    print("no photo has been already saved in one or both directories.")
                else:
                    numbers_left = [int(re.search(r'photo(\d+)\.jpg', f).group(1)) for f in existing_photos_left]
                    numbers_right = [int(re.search(r'photo(\d+)\.jpg', f).group(1)) for f in existing_photos_right]
                    if numbers_left == numbers_right:
                        photo_counter = max(numbers_left) + 1
                    else:
                        print("error: the number of photos for the right and the left cameras must be the same")
                        return 0
        else:
            if not os.path.exists(photo_directory_left):
                os.makedirs(photo_directory_left, exist_ok=True)
            else:
                shutil.rmtree(photo_directory_left)
                os.makedirs(photo_directory_left, exist_ok=True)

            if not os.path.exists(photo_directory_right):
                os.makedirs(photo_directory_right, exist_ok=True)
            else:
                shutil.rmtree(photo_directory_right)
                os.makedirs(photo_directory_right, exist_ok=True)

        cap_left=cv2.VideoCapture(camera_port_left)
        cap_right=cv2.VideoCapture(camera_port_right)
        cv2.namedWindow("Left Cam")
        cv2.namedWindow("Right Cam")
        print("Press 's' to capture a photo and 'q' to quit.")

        cap_time=time.time()
        old_time = time.time()
        while True:
            ret_left, org_frame_left = cap_left.read()
            ret_right, org_frame_right= cap_right.read()

            frame_left = org_frame_left.copy()
            frame_right = org_frame_right.copy()

            if not ret_left or not ret_right: 
                print(f"failed to grab frame from cameras or ports.")
                break
            frames = [frame_left,frame_right]
            old_time, frames = TwinCams.fps_calc(old_time, frames)
            frame_left = frames[0]
            frame_right= frames[1]

            if timer!=0:
                font                   = cv2.FONT_HERSHEY_DUPLEX
                bottomLeftCornerOfText = (20,200)
                fontScale              = 1
                fontColor              = (20,20,255)
                thickness              = 1
                lineType               = 2
                cv2.putText(frame_left, f"timer: {round(timer - (time.time()-cap_time) , 2)} s", bottomLeftCornerOfText, font,\
                fontScale, fontColor, thickness, lineType)
                cv2.putText(frame_right, f"timer: {round(timer - (time.time()-cap_time) , 2)} s", bottomLeftCornerOfText, font,\
                fontScale, fontColor, thickness, lineType)
            
            cv2.imshow("Left Cam", frame_left)
            cv2.imshow("Right Cam", frame_right)

            key = cv2.waitKey(1) & 0xFF

            if key in [ord('s'), ord('S')] or (timer - (time.time()-cap_time) < 0 and timer>0):  # take photo after pressing "s" or timer is activated
                filePath = os.path.join(photo_directory_left, f"photo{photo_counter}.jpg") 
                cv2.imwrite(filePath, org_frame_left)
                print(f"Photo saved in {filePath}.")
                filePath = os.path.join(photo_directory_right, f"photo{photo_counter}.jpg") 
                cv2.imwrite(filePath, org_frame_right)
                print(f"Photo saved in {filePath}.")
                photo_counter=photo_counter+1
                cap_time = time.time()
                if timer>0:
                    TwinCams.do_beep()

            if key in [ord('q'), ord('Q')] or (photo_counter>max_number and max_number>0):  # exits if "q" is pressed or the limit is reached
                print("Exiting function.")
                break            

        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
    # *****************************************************************************************************************************
    @staticmethod
    def fps_calc(old_time, frames):
        font                   = cv2.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = (20,30)
        fontScale              = 1
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = 2

        new_time=time.time()
        deltaT=new_time-old_time
        fps=1/deltaT
        fps=int(fps)

        if isinstance(frames, list):  # Check if frames is a list
            for frame in frames:
                cv2.putText(frame, f"fps: {fps}",
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
        else:  # Single frame
            cv2.putText(frames, f"fps: {fps}",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

        return new_time, frames
    # *****************************************************************************************************************************
    @staticmethod
    def stereo_calibrator (square_width=37.5, relative_path= "Photos", length_unit="mm",chessboard_column=9, chessboard_row=6, max_iter=300, eps=0.0001,\
                            max_iter_cal=100, eps_cal=1e-5, search_window_size=(5, 5), result_path='stereo_calibration.pkl' ): # this function is written based on https://github.com/bvnayak/stereo_calibration/blob/master/camera_calibrate.py
                                                                   # for column and row numbers look at your chessboard, 
                                                                   # they are usually number of horizental and vertical squares -1
                                                                   # square_width=37.43
        # Calibration setup:
        criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps) # for your information "+" sign works as "OR" operation for criteria
        criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, max_iter_cal, eps_cal)
        objp = np.zeros((chessboard_column*chessboard_row, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_column, 0:chessboard_row].T.reshape(-1, 2)* square_width
        objpoints = []  # 3d point in real world space
        imgpoints_l = []  # 2d points in image plane.
        imgpoints_r = []  # 2d points in image plane.     

        # Check images folders:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        photo_directory_left = os.path.join(base_dir, relative_path, "Left")
        photo_directory_right = os.path.join(base_dir, relative_path, "Right")

        if not os.path.exists(photo_directory_left):
            print("No left images folder found")
            return 0
        if not os.path.exists(photo_directory_right):
            print("No right images folder found")
            return 0
        
        else:
            existing_photos_left = [f for f in os.listdir(photo_directory_left) if re.match(r'photo(\d+)\.jpg', f)]
            existing_photos_right = [f for f in os.listdir(photo_directory_right) if re.match(r'photo(\d+)\.jpg', f)]
            if not existing_photos_left or not existing_photos_right:
                print("no photo has been already saved in one or both directories.")
                return 0

        images_right = glob.glob(relative_path + '/Right/*.jpg')
        images_left = glob.glob(relative_path + '/Left/*.jpg')
            
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (chessboard_column, chessboard_row), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (chessboard_column, chessboard_row), None)       

            if ret_l is True and ret_r is True:
                # ####### left :
                rt = cv2.cornerSubPix(gray_l, corners_l, search_window_size,
                                      (-1, -1), criteria)                 
                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (chessboard_column, chessboard_row),
                                                  corners_l, ret_l)
                cv2.imshow("left cam image", img_l)

                # ####### right:
                rt = cv2.cornerSubPix(gray_r, corners_r, search_window_size,
                                      (-1, -1), criteria)
                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r,(chessboard_column, chessboard_row),
                                                  corners_r, ret_r)
                cv2.imshow("right cam image", img_r)

                key=cv2.waitKey(0)                
                if key in [ord('q'), ord('Q')]: # q stands for skip
                    pass
                if key in [ord('s'), ord('S')]: # s for save
                    imgpoints_r.append(corners_r)
                    imgpoints_l.append(corners_l)
                    objpoints.append(objp)
                cv2.destroyAllWindows()

            img_shape = gray_l.shape[::-1]

        rt, M1, d1, r1, t1 = cv2.calibrateCamera(
            objpoints, imgpoints_l, img_shape, None, None)
        rt, M2, d2, r2, t2 = cv2.calibrateCamera(
            objpoints, imgpoints_r, img_shape, None, None)

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        
        dims = img_shape
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, max_iter_cal, eps_cal)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l,
            imgpoints_r, M1, d1, M2,
            d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', r1),
                            ('rvecs2', r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        with open(result_path, 'wb') as f:
            pickle.dump({'M1': M1, 'M2': M2, 'd1': d1, 'd2': d2, 'R': R, 'T': T}, f)

        cv2.destroyAllWindows()
    # *****************************************************************************************************************************
    def load_calibration_result(result_path='stereo_calibration.pkl' ): # tihs function reads values from calibration result file
                                                                        # and calculates needed variables
        with open(result_path, 'rb') as f:
            calibration_data = pickle.load(f)
        M1 = calibration_data['M1']
        M2 = calibration_data['M2']
        d1 = calibration_data['d1']
        d2 = calibration_data['d2']
        R = calibration_data['R']
        T = calibration_data['T']

        RT1 = np.c_[np.eye(3), np.zeros((3, 1))]
        RT2 = np.c_[R, T]

        P1 = M1 @ RT1
        P2 = M2 @ RT2
        return M1, M2, d1, d2, P1, P2
    # *****************************************************************************************************************************
    def point_3d_calc (point1, point2, M1, M2, d1, d2, P1, P2):
        undist_point1= cv2.undistortPoints(point1, M1, d1, None, M1)
        undist_point2= cv2.undistortPoints(point2, M2, d2, None, M2)
        point_4d = cv2.triangulatePoints(P1 ,P2, undist_point1, undist_point2)
        point_3d = point_4d[:3]/point_4d[3]

        return point_3d
    # *****************************************************************************************************************************
    def do_beep(frequency = 300, duration= 0.5, sample_rate=44100): # this function create a sound for auto capturing mode
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave_data = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

        sd.play(wave_data, sample_rate)
        sd.wait()  
    # *****************************************************************************************************************************
    def return_3d_point (relative_path): # this function gets path of photos from two cameras and returns a 3d point based on the inputs of the client
        global click_count1
        global click_count2
        global point1_webcam1, point1_webcam2

        point1_webcam1 = None
        point1_webcam2 = None

        # to handle mouse click events
        click_count1 = 0
        click_count2=0
        
        # a function to handle clicking:
        def click_event(event, x, y, flags, param):
            global click_count1
            global click_count2
            global point1_webcam1, point1_webcam2

            if event == cv2.EVENT_LBUTTONDOWN:
                if param == 1:
                    point1_webcam1 = (x, y)
                    print(f"Webcam 1 - Point 1: ({x}, {y})")
                    click_count1 += 1

                elif param == 2:
                    point1_webcam2 = (x, y)
                    print(f"Webcam 2 - Point 1: ({x}, {y})")
                    click_count2 += 1
        

        # Check images folders:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        photo_directory_left = os.path.join(base_dir, relative_path, "Left")
        photo_directory_right = os.path.join(base_dir, relative_path, "Right")

        if not os.path.exists(photo_directory_left):
            print("No left images folder found")
            return 0
        if not os.path.exists(photo_directory_right):
            print("No right images folder found")
            return 0
        
        else:
            existing_photos_left = [f for f in os.listdir(photo_directory_left) if re.match(r'photo(\d+)\.jpg', f)]
            existing_photos_right = [f for f in os.listdir(photo_directory_right) if re.match(r'photo(\d+)\.jpg', f)]
            if not existing_photos_left or not existing_photos_right:
                print("no photo has been already saved in one or both directories.")
                return 0


        images_right = glob.glob(relative_path + '/Right/*.jpg')
        images_left = glob.glob(relative_path + '/Left/*.jpg')

        images_left.sort()
        images_right.sort()

        M1, M2, d1, d2, P1, P2 = TwinCams.load_calibration_result()
        # Set up windows for the webcam outputs
        
        cv2.namedWindow("Webcam 1")
        cv2.setMouseCallback("Webcam 1", click_event, param=1)

        cv2.namedWindow("Webcam 2")
        cv2.setMouseCallback("Webcam 2", click_event, param=2)  
        calculated_3d_points=[]
        
        for i, fname in enumerate(images_right):
            while True:
                frame_left = cv2.imread(images_left[i])
                frame_right = cv2.imread(images_right[i])
                # Draw circles on clicked points
                if point1_webcam1:
                    cv2.circle(frame_left, point1_webcam1, 5, (0, 255, 0), -1)  # Green circle on webcam 1
                if point1_webcam2:
                    cv2.circle(frame_right, point1_webcam2, 5, (0, 0, 255), -1)  # Red circle on webcam 2

                cv2.imshow("Webcam 1", frame_left)
                cv2.imshow("Webcam 2", frame_right)

                if point1_webcam1 and point1_webcam2:

                    points3D_1 = TwinCams.point_3d_calc(point1_webcam1, point1_webcam2, M1, M2, d1, d2, P1, P2)
                    calculated_3d_points.append(points3D_1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    midpoint1 = (point1_webcam1[0] , point1_webcam1[1])
                    midpoint2 = (point1_webcam2[0], point1_webcam2[1] )
                    font_scale = 1
                    font_color = (255, 0, 0)  # Blue color in BGR
                    font_thickness = 2
                    number = np.array2string((np.round(points3D_1,0)))
                    cv2.putText(frame_left, number, midpoint1, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                    cv2.putText(frame_right, number, midpoint2, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                    cv2.imshow("Webcam 1", frame_left)
                    cv2.imshow("Webcam 2", frame_right)
                    cv2.waitKey(3000)
                    break
                # Wait for keypress to exit the loop
                key= cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q')]:  # Press 'q' to quit
                    break
        cv2.destroyAllWindows()
        return calculated_3d_points
    # *****************************************************************************************************************************
    def distance_calc (ports):
        global click_count1
        global click_count2
        global point1_webcam1, point2_webcam1, point1_webcam2, point2_webcam2

        # choosen points are saved in these variables:
        point1_webcam1 = None
        point2_webcam1 = None
        point1_webcam2 = None
        point2_webcam2 = None

        # to handle mouse click events
        click_count1 = 0
        click_count2=0

        # a function to handle clicking:
        def click_event(event, x, y, flags, param):
            global click_count1
            global click_count2
            global point1_webcam1, point2_webcam1, point1_webcam2, point2_webcam2

            if event == cv2.EVENT_LBUTTONDOWN:
                if param == 1:
                    if click_count1%2==0:
                        point1_webcam1 = (x, y)
                        print(f"Webcam 1 - Point 1: ({x}, {y})")
                    else:
                        point2_webcam1 = (x, y)
                        print(f"Webcam 1 - Point 2: ({x}, {y})")
                    click_count1 += 1

                elif param == 2:
                    if click_count2%2 == 0:
                        point1_webcam2 = (x, y)
                        print(f"Webcam 2 - Point 1: ({x}, {y})")
                    else:
                        point2_webcam2 = (x, y)
                        print(f"Webcam 2 - Point 2: ({x}, {y})")
                    click_count2 += 1

        M1, M2, d1, d2, P1, P2 = TwinCams.load_calibration_result()
        camera_port_left= ports[0]
        camera_port_right= ports[1]

        cap_left=cv2.VideoCapture(camera_port_left)
        cap_right=cv2.VideoCapture(camera_port_right)

        if not cap_left.isOpened() or not cap_right.isOpened():
            print("Error: Could not open one or both cameras.")
            exit()
        # Set up windows for the webcam outputs
        cv2.namedWindow("Webcam 1")
        cv2.setMouseCallback("Webcam 1", click_event, param=1)

        cv2.namedWindow("Webcam 2")
        cv2.setMouseCallback("Webcam 2", click_event, param=2)  

        while True:
            ret_left, org_frame_left = cap_left.read()
            ret_right, org_frame_right= cap_right.read()

            frame_left = org_frame_left.copy()
            frame_right = org_frame_right.copy()

            if not ret_left or not ret_right: 
                print(f"failed to grab frame from cameras or ports.")
                break
            
            # Draw circles on clicked points
            if point1_webcam1:
                cv2.circle(frame_left, point1_webcam1, 5, (0, 255, 0), -1)  # Green circle on webcam 1
            if point2_webcam1:
                cv2.circle(frame_left, point2_webcam1, 5, (0, 255, 0), -1)  # Green circle on webcam 1
            if point1_webcam2:
                cv2.circle(frame_right, point1_webcam2, 5, (0, 0, 255), -1)  # Red circle on webcam 2
            if point2_webcam2:
                cv2.circle(frame_right, point2_webcam2, 5, (0, 0, 255), -1)  # Red circle on webcam 2
            cv2.imshow("Webcam 1", frame_left)
            cv2.imshow("Webcam 2", frame_right)

            if point1_webcam1 and point1_webcam2 and point2_webcam1 and point2_webcam2:

                points3D_1 = TwinCams.point_3d_calc(point1_webcam1, point1_webcam2, M1, M2, d1, d2, P1, P2)
                points3D_2 = TwinCams.point_3d_calc(point2_webcam1, point2_webcam2, M1, M2, d1, d2, P1, P2)
         
                distance=TwinCams.euclidean_distance(points3D_1,points3D_2)
                print("with undistortPoints:",distance)

                line_thickness = 2
                cv2.line(frame_left, point1_webcam1, point2_webcam1, (0, 255, 0), line_thickness)
                cv2.line(frame_right, point1_webcam2, point2_webcam2, (0, 0, 255), line_thickness)

                font = cv2.FONT_HERSHEY_SIMPLEX
                midpoint1 = ((point1_webcam1[0] + point2_webcam1[0]) // 2, (point1_webcam1[1] + point2_webcam1[1])// 2)
                midpoint2 = ((point1_webcam2[0] + point2_webcam2[0]) // 2, (point1_webcam2[1] + point2_webcam2[1])// 2)
                font_scale = 1
                font_color = (255, 0, 0)  # Blue color in BGR
                font_thickness = 2
                number = str(round(distance,0))
                cv2.putText(frame_left, number, midpoint1, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                cv2.putText(frame_right, number, midpoint2, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                cv2.imshow("Webcam 1", frame_left)
                cv2.imshow("Webcam 2", frame_right)
                cv2.waitKey(10000)
                quit()

            # Wait for keypress to exit the loop
            key= cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:  # Press 'q' to quit
                break
        
        # Print the stored coordinates
        print(f"Final Coordinates for Webcam 1: Point 1: {point1_webcam1}, Point 2: {point2_webcam1}")
        print(f"Final Coordinates for Webcam 2: Point 1: {point1_webcam2}, Point 2: {point2_webcam2}")

        # Release resources and close windows
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()        
    # *****************************************************************************************************************************
    def euclidean_distance(point1, point2):
        if len(point1) != len(point2):
            raise ValueError("points must have the same dimensions")
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.linalg.norm(point1 - point2)
        
        

    # *****************************************************************************************************************************
if __name__ == "__main__": 
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # a summary about functions: 
        # TwinCams.find_camera_ports --> this function shows you ports number of your webcams/cameras.
        # TwinCams.cameras_orientation --> It is very important that the order of cameras in calibration and other parts of your project be the same. This one returns ports in (left, right) order
        # TwinCams.single_cam_take_photo --> takes photo upon you press "s" key and save it. click "q" to exit. Also you can use timer and set the number of photos to take pictures automatically.
        # TwinCams.stereo_cam_take_photo --> same as previous one but for stereo setup. note that for stereo calibration you need to take pictures for both cameras at the same time.
        # TwinCams.fps_calc --> this is used within other function to calcualte fps of your camera/cameras stream
        # TwinCams.stereo_calibrator --> after taking pictures, use this one. press "q" if chessboard is not shown well, otherwise press "s". Dont forget to change these parameters according to your setup:
                                                                                            # square_width, length_unit,chessboard_column, chessboard_row
        # TwinCams.load_calibration_result --> load calibration results from the previous function and also does some calculations
        # TwinCams.return_3d_point --> this one gets a relative path of a folder which contains left and rights phots, then open photos and you can get points in 3d by clicking on the same points on the two frames
        # TwinCams.point_3d_calc --> calculates the point which is represented in the view of each cameras separately in the general coordinate system in 3d coordinates
        # TwinCams.do_beep --> just generates a beep and it is used in auto capturing mode
        # TwinCams.distance_calc --> you can click on the points you want to calculate the distance between two points in the real world. note that the order of clicking on the points on each
                                                                                                # window is very important.
        # TwinCams.euclidean_distance --> use to calculate distance between two points.
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    ports=TwinCams.cameras_orientation(2,4, stream = True)
    # TwinCams.stereo_cam_take_photo((2,4), timer=5, max_number=8)
    # TwinCams.stereo_calibrator(search_window_size=(4,4))
    # TwinCams.distance_calc((2,4))
    
