import cv2
import numpy as np


arucodict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

param_markers=cv2.aruco.DetectorParameters()
cap=cv2.VideoCapture(0)

intrinsic_camera=np.array(((857.99152457,0.0,322.16590173),(0.0,852.889,273.573),(0.0,0.0,1.0)))
distortion=np.array((-0.466,0.00335,-0.0074,0))
while True:
    ret,frame=cap.read()
    if not ret:
        break
    gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector=cv2.aruco.ArucoDetector(arucodict,param_markers)
    
    marker_corner,marker_ids,reject=  detector.detectMarkers(gray_frame)
    print(marker_ids)
    totalmarker=0
    if marker_ids is not None:
        for c in marker_corner:
            marker_points=np.array([[-2/2,2/2,0],
                                    [2/2,2/2,0],
                                    [2/2,-2/2,0],
                                    [-2/2,-2/2,0]],dtype=np.float32)
            markerpoints,rvec,tvec=cv2.solvePnP(marker_points,c,intrinsic_camera,distortion,False,cv2.SOLVEPNP_IPPE_SQUARE)
            totalmarker=range(0,marker_ids.size)
            cv2.aruco.drawDetectedMarkers(frame,marker_corner,marker_ids)
            frame=cv2.drawFrameAxes(frame,intrinsic_camera,distortion,rvec,tvec,length=0.4,thickness=2)
            cv2.putText(frame,
                        "distance",
                        (40,50),
                        cv2.FONT_HERSHEY_PLAIN, 
                        1.3, 
                        (200,100,255),
                        2)
            
            cv2.putText(frame,
                        str(round(tvec[2][0],2)),
                        (150,50),
                        cv2.FONT_HERSHEY_PLAIN, 
                        1.3, 
                        (200,100,255),
                        2)
            cv2.putText(frame,
                        "X:",
                        (100,400),
                        cv2.FONT_HERSHEY_PLAIN, 
                        1.3, 
                        (200,100,255),
                        2)
            cv2.putText(frame,
                        str(round(tvec[0][0],2)),
                        (120,400),
                        cv2.FONT_HERSHEY_PLAIN, 
                        1.3, 
                        (200,100,255),
                        2)
            cv2.putText(frame,
                        "Y:",
                        (180,400),
                        cv2.FONT_HERSHEY_PLAIN, 
                        1.3, 
                        (200,100,255),
                        2)
            cv2.putText(frame,
                        str(round(tvec[1][0],2)),
                        (200,400),
                        cv2.FONT_HERSHEY_PLAIN, 
                        1.3, 
                        (200,100,255),
                        2)
            
            
        
    
    
    
    
    
    
    cv2.imshow("frame", frame)
    k=cv2.waitKey(1)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()