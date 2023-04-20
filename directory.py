import cv2
from tensorflow.keras.models import load_model
import numpy as np
import winsound
import os
import time
from twilio.rest import Client
# Load the saved model
loaded_model = load_model('my_model.h5')

# Define the emotion labels
emotion_labels = ['Not Attentive', 'Not Attentive', 'Not Attentive', 'Not Attentive', 'Not Attentive', 'Not Attentive', 'Attentive']

# Create a face detector object
face_detector = cv2.CascadeClassifier('C:/DOCUMENTS/8th sem/Project/data/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Set the threshold for attentiveness
threshold = 0.5

# Initialize counters for attentive and not attentive faces
attentive_faces = 0
not_attentive_faces = 0

last_attentive_time = time.time()
# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # Update the total faces counter
    total_faces = len(faces)

    # Reset the counters for attentive and not attentive faces
    attentive_faces = 0
    not_attentive_faces = 0

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face = frame[y:y+h, x:x+w]

        # Preprocess the face image
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = np.reshape(face, [1, 48, 48, 1])

        # Predict the emotion of the face
        result = loaded_model.predict(face)[0]

        # Determine the predicted emotion label
        predicted_emotion = emotion_labels[np.argmax(result)]

        # Draw a rectangle around the face
        if predicted_emotion == 'Attentive':
            color = (0, 255, 0)  # green
            attentive_faces += 1
        else:
            color = (0, 0, 255)  # red
            not_attentive_faces += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Add the predicted emotion label to the frame
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Check if any faces were detected
    if total_faces > 0:
        # Calculate the percentage of attentive faces
        percent_attentive = (attentive_faces / total_faces)*100

        # Check if the percentage of attentive faces is below the threshold
        if percent_attentive < threshold:
            # Play a sound to alert the user
            if time.time() - last_attentive_time > 10:
                # Play a sound to alert the user
                winsound.Beep(1000, 500)
                account_sid = "Enter your SID NUMBER"
                auth_token = "Enter your Auth token"
                client = Client(account_sid, auth_token)
                message = client.messages.create(
                    body= percent_attentive,
                    from_="default phone number",
                    to="phonenumber"
                )
                print(message.sid)

    # Show the frame
        cv2.imshow('frame', frame)
        print('Percentage of attentive faces:',percent_attentive,'%')
        print("number of faces:",total_faces)
    # Check if the 'q' key was pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
