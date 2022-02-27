import cv2
import glob
import os
from cv2 import CascadeClassifier
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('saved_mask_model.h5')

cap = cv2.VideoCapture(0)
face_cascade = CascadeClassifier('haarcascade_frontalface_default.xml')
img_wt, img_ht = 150, 150

while True:
    img_count = 0
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = img[y:y+h, x:x+w]
        org = (x-10, y-10)
        img_count += 1

        cv2.imwrite('input\\face%d.jpg' % (img_count), roi_color)
        color = image.load_img('input\\face%d.jpg' %
                               (img_count), target_size=(img_wt, img_ht))
        images = image.img_to_array(color)  # to convert to numpy array
        images = np.expand_dims(images, axis=0)
        prediction = model.predict(images)

        if prediction == 0:
            label = 'Mask'
            color = (0, 255, 0)
        else:
            label = 'No Mask'
            color = (0, 0, 255)

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 3, cv2.LINE_AA)

    cv2.imshow('Face Detection', img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        files = glob.glob('input/*')
        for f in files:
            os.remove(f)
        break

cap.release()
cv2.destroyAllWindows()
