from flask import Flask, request, jsonify
import numpy as np
import cv2 as cv
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials, storage as admin_storage
import requests
import tempfile
import os

cred = credentials.Certificate(
    "app/surfen-c0856-firebase-adminsdk-cnjfu-c926b6a0b5.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'surfen-c0856.appspot.com'
})
app = Flask(__name__)


@app.route('/api/process-video', methods=['POST'])
def process_video():
    video_url = request.args.get('video_url')
    video_bytes = download_video(video_url)
    processed_video_bytes = optical_process(video_bytes)
    # SUCCESS CODE 200
    return jsonify({'processed_video_url': processed_video_bytes}), 200

def download_video(url):
    response = requests.get(url)
    return response.content


def optical_process(video_bytes):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.write(video_bytes)
    temp_file.close()

    # OUT FILE
    temp_file_out = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    temp_path_out = temp_file_out.name
    temp_file_out.close()

    video = cv.VideoCapture(temp_file.name)

    # OUTPUT DIMENSIONS
    out_w, out_h = 350, 350 
    out = cv.VideoWriter(temp_path_out,
                         cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (out_w, out_h))
    
    # LUKAS-KANADE PARAMS
    feature_params = dict(maxCorners=1,
                          qualityLevel=0.5,
                          minDistance=5,
                          blockSize=15)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # FIND FIRST CORNERS
    ret, old_frame = video.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    while (1):
        ret, frame = video.read()
        if not ret:
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # OPTICAL FLOW DIRECTION
        p1, st, err = cv.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params)
        # SELECT GOOD POINTS
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        # DRAW TRACKS
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

        # EXTRACTING WINDOW OF SURFER
        out_window = cv.getRectSubPix(
            frame, (out_w, out_h), (int(a) - 75, int(b) - 75))

        # SHARPEN
        # sharpen_kernel = np.array([[-0.8, -1, -0.8],
        #                            [-1,  9, -1],
        #                            [-0.8, -1, -0.8]])
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        out_window = cv.filter2D(out_window, -1, sharpen_kernel)

        # FILTER
        # out_window = cv.medianBlur(out_window, 11)

        # show and write
        out.write(out_window)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    print(f"Size of temporary file: {os.stat(temp_path_out).st_size} bytes")
    video.release()
    out.release()

    # Define the Firebase bucket
    bucket = admin_storage.bucket()

    # Create a blob and upload the BytesIO object
    blob = bucket.blob('output_video_4.avi')

    with open(temp_path_out, 'rb') as tf:
        blob.upload_from_file(tf, content_type='video/avi')

    # Make the blob publicly accessible and get the public URL
    blob.make_public()
    public_url = blob.public_url

    # DELETE TEMP FILE
    os.unlink(temp_path_out)

    return public_url


port = int(os.environ.get('PORT', 5000))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)


