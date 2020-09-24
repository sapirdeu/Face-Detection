import cv2
#from random import randrange

# load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
img = cv2.imread('Donald_Melania.jpg')

# Convert to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

#print(face_coordinates)

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with the faces spotted
cv2.imshow('Donald and Melania Trump', img)

# wait here in the code and listen for a key press
cv2.waitKey()