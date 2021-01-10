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
