import cv2
import numpy as np
import urllib.request
import imghdr
import tempfile
import requests
import io


# Get video from URL
url = "https://storage.googleapis.com/surfen-c0856.appspot.com/output_video.mp4"
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

response = requests.get(url)
video_bytes = io.BytesIO(response.content).read()


temp_file.write(video_bytes)
temp_file.close()

# Open the video
cap = cv2.VideoCapture(temp_file.name)

# Check if video opened successfully
if (cap.isOpened() == False):
    print("Unable to read video")

# Read until video is completed
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
