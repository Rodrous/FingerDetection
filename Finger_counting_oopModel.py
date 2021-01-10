import numpy as np
import cv2 as cv

class fingerdetection:

    def masking(self,image):
        hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([20, 255, 255], dtype = "uint8")
        skinRegionHSV = cv.inRange(hsvim, lower, upper)
        blurred = cv.blur(skinRegionHSV, (2,2))
        _, thresholded = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
        cv.imshow("mask", skinRegionHSV)
        return thresholded

    def redefine_cont_hull(self,masked_image):
        contours, _ = cv.findContours(masked_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = max(contours, key=lambda x: cv.contourArea(x))
        hull = cv.convexHull(contours)
        return contours, hull

    def get_defects(self,contours):
        hull = cv.convexHull(contours, returnPoints=False)
        defects = cv.convexityDefects(contours, hull)
        return defects

    def main(self):
         #Image Capture, for video set it to 0
         cap = cv.VideoCapture('four.jpg')
         while cap.isOpened():
            _,img = cap.read()
            try:
                mask_image = self.masking(img)
                contours,hull = self.redefine_cont_hull(mask_img)
                cv.drawContours(img, [contours], -1, (255,255,0), 2)
                cv.drawContours(img, [hull], -1, (0, 255, 255), 2)

                defects = self.get_defects(contours)
                if defects is not None:
                    cnt = 0
                    for i in range(defects.shape[0]):
                        startpoint,endpoint,farthest,approximate = defects[i][0]
                        start = tuple(contours[startpoint][0])
                        end = tuple(contours[endpoint][0])
                        far = tuple(contours[farthest][0])
                        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                        avg = (a+b+c)/2
                        one = np.sqrt(avg*(avg-a)*(avg-b)*(avg-c))
                        distance_pc = (2*one)/a

                        if angle<= np.pi/2 and distance_pc> 30:
                            cnt +=1
                            cv.circle(img,far,4,[0,0,255],-1)
                    if cnt > 0:
                        cnt = cnt+1
                    cv.putText(img,str(cnt),(0,50),cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
                cv.imshow("img",img)

            except:
                pass
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
         cap.release()
         cv.destroyAllWindows()

initial = fingerdetection()
initial.main()
