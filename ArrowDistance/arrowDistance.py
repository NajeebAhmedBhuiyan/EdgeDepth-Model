import time
import cv2
import numpy as np
import math

# Dummy Motor class for later use in Rpi5
class Motor:
    def __init__(self, *args):
        pass
    def forward(self, *args, **kwargs):
        pass
    def reverse(self, *args, **kwargs):
        pass
    def stop(self, *args, **kwargs):
        pass

# Image stacking utility
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: 
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: 
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

# Arrow detection with distance measurement
def getContours(inImg, outImg):
    contours, _ = cv2.findContours(inImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            cv2.drawContours(outImg, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)           
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if len(approx) == 7:
                # Get bounding box coordinates
                x, y, w, h = cv2.boundingRect(approx)
                
                # Calculate diagonal length of bounding box
                diagonal_length = math.sqrt(w**2 + h**2)
                
                # Draw bounding box and display distance
                cv2.rectangle(outImg, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.drawContours(outImg, [approx], -1, (0, 255, 0), 2)
                
                # Display distance at top-left of bounding box
                cv2.putText(outImg, f"{int(diagonal_length)}", 
                           (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0), 2)
                
                # Determine arrow direction
                _, _, angle = cv2.fitEllipse(approx)
                if 80 < angle < 100:
                    xval = list(approx[:, 0, 0])
                    arrow_center = (max(xval) + min(xval)) / 2
                    
                    # Print direction and distance in terminal
                    if np.median(xval) < arrow_center:
                        print(f"LEFT ARROW DETECTED - Distance: {int(diagonal_length)}")
                    else:
                        print(f"RIGHT ARROW DETECTED - Distance: {int(diagonal_length)}")

# Initialize simulated motors
motor1 = Motor(13, 26, 19)
motor2 = Motor(16, 21, 20)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 720)  # Width
cap.set(4, 405)  # Height
kernel = np.ones((5, 5), np.uint8)

# Variable to track last print time
last_print_time = 0
print_interval = 0.5  # seconds

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            break
            
        imgOut = img.copy()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
        imgCanny = cv2.Canny(imgBlur, 30, 100)
        imgDilated = cv2.dilate(imgCanny, kernel, iterations=2)
        imgEroded = cv2.erode(imgDilated, kernel, iterations=1)
        
        getContours(imgEroded, imgOut)
        
        # Create image grid
        imgStack = stackImages(0.5, ([img, imgGray, imgOut],
                                  [imgCanny, imgDilated, imgEroded]))
        
        cv2.imshow("Arrow Recognition with Distance Measurement", imgStack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except Exception as ex:
    print(f"Error: {ex}")
    
finally:
    cap.release()
    cv2.destroyAllWindows()