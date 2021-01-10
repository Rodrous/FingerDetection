import numpy as np
import cv2 as cv

def masking(img):
    hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv.inRange(hsvim, lower, upper)
    blurred = cv.blur(skinRegionHSV, (2,2))
    _, thresholded = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
    cv.imshow("mask", skinRegionHSV)
    return thresholded

def read_cont_hull(mask_img):
    contours, _ = cv.findContours(mask_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv.contourArea(x))
    hull = cv.convexHull(contours)
    return contours, hull

def getdefects(contours):
    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)
    return defects

cap = cv.VideoCapture(0) 
while cap.isOpened():
    _, img = cap.read()
    try:
        mask_img = masking(img)
        contours, hull = read_cont_hull(mask_img)
        cv.drawContours(img, [contours], -1, (255,255,0), 2)
        cv.drawContours(img, [hull], -1, (0, 255, 255), 2)
        defects = getdefects(contours)
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):
                startpoint, endpoint, farthest, approximate = defects[i][0]
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

                if angle <= np.pi / 2 and distance_pc>30:
                    cnt += 1
                    cv.circle(img, far, 4, [0, 0, 255], -1)
            if cnt > 0:
                cnt = cnt+1
            cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
        cv.imshow("img", img)

    except:
        pass
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

#
#Gesutre Implementation, NN
def tensor():
    from tensorflow.keras.models import load_model
    model = load_model('keras_model.h5',compile= False)
    import cv2
    import numpy as np
    import time

    prediction = ''
    action = ''
    score = 0
    img_counter = 500

    gesture_names = {0:'bird',
        1: 'boar',
        2: 'dog',
        3: 'dragon',
        4: 'hare',
        5: 'horse',
        6: 'monkey',
        7: 'ox',
        8: 'ram',
        9: 'rat',
        10: 'snake',
        11: 'Tiger',
        12: 'Zero'}

def predict_image(image):

    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)

    # model.predict() returns an array of probabilities -
    # np.argmax grabs the index of the highest probability.
    result = gesture_names[np.argmax(pred_array)]

    # The score is a float, but I wanted to
    # display just 2 digits beyond the decimal point.
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(f'Result: {result}, Score: {score}')
    return result, score

def applied_model():
    import time
    if cv2.waitKey(1) & 0xFF == ord('c'):
        resize = cv2.resize(img (224, 224))
        reshape = resize.reshape(1,224,224,3)
        prediction, score = predict_image(reshape)
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        exit

    else:
        pass
