'''import cv2

# Pre-trained Haar Cascade for face/object detection using standard OpenCV built-ins
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

print("Press 'q' to quit...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Target Object (Face)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Robot Vision: Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
#!pip install --force-reinstall opencv-python

import cv2

# Pre-trained Haar Cascade for face/object detection using standard OpenCV built-ins
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Note: cv2.VideoCapture(0) and cv2.imshow may not work directly on a remote Google Colab server 
# as it lacks access to your local webcam and display GUI. 
# If running locally, this will work. 
cap = cv2.VideoCapture(0)

print("Press 'q' to quit...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Target Object (Face)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # If running in Colab, you can replace cv2.imshow with google.colab.patches.cv2_imshow
    try:
        cv2.imshow("Robot Vision: Object Detection", frame)
    except cv2.error:
        from google.colab.patches import cv2_imshow
        cv2_imshow(frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()