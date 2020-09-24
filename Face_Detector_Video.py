import cv2
from random import randrange

# load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture a video from webcam
#webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
webcam = cv2.VideoCapture('giphy.gif')

# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Convert to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
    
    # Display the image with the faces spotted
    cv2.imshow('Camera Frame', frame)
    
    # wait here in the code and listen for a key press
    key = cv2.waitKey(1)

    # Stop if Q or q is pressed
    if key==81 or key==113:
        break

# Release the VideoCapture object
webcam.release()