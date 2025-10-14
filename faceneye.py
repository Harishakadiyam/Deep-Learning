import numpy as np
import cv2
            
face_cascade = cv2.CascadeClassifier(r'C:\Users\Admin\AVScode\AI\Haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'C:\Users\Admin\AVScode\AI\Haarcascades\haarcascade_eye.xml')

image = cv2.imread(r'C:\Users\Admin\Downloads\hrithik.jpeg')

# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or cannot be loaded!")
    exit()  # Exit if image is not loaded
    
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)
    
        

roi_gray = gray[y:y+h, x:x+w]
roi_color = image[y:y+h, x:x+w]
eyes = eye_cascade.detectMultiScale(roi_gray)
for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        eye_cascade.detectMultiScale(gray)

        

cv2.imshow('Face and Eye Detection', image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
    