from flask import Flask, request, jsonify
import numpy as np
import cv2 as cv
import io
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials, storage as admin_storage
import requests
import tempfile
import imageio
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

    # 2. Download the video
    video_bytes = download_video(video_url)

    # 3. Process the video
    processed_video_bytes = optical_process(video_bytes)

    # 4. Upload the video
    # processed_video_url = upload_video(processed_video_bytes)

    # 5. Return the processed video url

    response = {'video_url': video_url}

    return jsonify({'processed_video_url': processed_video_bytes}), 200
    # return response


def download_video(url):
    # Get the bucket and blob from the URL
    # Format: gs://bucket_name/blob_name
    response = requests.get(url)
    return response.content


def optical_process(video_bytes):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.write(video_bytes)
    temp_file.close()

    # OUT
    temp_file_out = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    temp_path_out = temp_file_out.name
    temp_file_out.close()

    ###

    video = cv.VideoCapture(temp_file.name)

    out_w, out_h = 350, 350  # output dimensions - adjust as needed
    out = cv.VideoWriter(temp_path_out,
                         cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (out_w, out_h))

    feature_params = dict(maxCorners=1,
                          qualityLevel=0.5,
                          minDistance=5,
                          blockSize=15)

    # Lucas Kanade optical flow params
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Creating random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = video.read()

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while (1):
        ret, frame = video.read()
        if not ret:
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
        #     mask = cv.line(mask, (int(a), int(b)),
        #                    (int(c), int(d)), color[i].tolist(), 2)
        #     frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        # img = cv.add(frame, mask)
        # cv.imshow('frame', img)

        # img2 = cv.rectangle(frame, (int(a) - 50, int(b) - 50),
        #                     (int(a) + 50, int(b)+50), 255, 2)
        # # cv.imshow('img2', img2)

        # EXTRACTING WINDOW OF SURFER
        out_window = cv.getRectSubPix(
            frame, (out_w, out_h), (int(a) - 75, int(b) - 75))

        # sharpen
        # sharpen_kernel = np.array([[-0.8, -1, -0.8],
        #                            [-1,  9, -1],
        #                            [-0.8, -1, -0.8]])
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        out_window = cv.filter2D(out_window, -1, sharpen_kernel)

        # filter
        # out_window = cv.medianBlur(out_window, 11)

        # show and write
        out.write(out_window)

        # Unsharp Masking (USM): This method involves creating a blurred version of the image and subtracting it from the original image to create a sharpened version. It can be effective for sharpening small areas in an image.
        # High Pass Filtering: This method involves removing the low-frequency components of an image and retaining the high-frequency components, which can enhance details and edges. This can be effective for sharpening small areas in an image.
        # Local Contrast Enhancement: This method involves increasing the contrast in small areas of an image, which can make edges and details more visible. This can be done using tools like the "Clarity" slider in Adobe Lightroom.
        # Selective Sharpening: This method involves using a brush or selection tool to apply sharpening only to specific areas of an image that need it. This can be effective for sharpening small areas in an image while minimizing the impact on other parts of the image.

        # TRYING BasicVSR++

        # key = cv.waitKey(0) & 0xFF
        # if key == ord('n'):
        #     continue
        # k = cv.waitKey(30) & 0xff
        # if k == ord('q'):
        #     break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    print(f"Size of temporary file: {os.stat(temp_path_out).st_size} bytes")

    video.release()
    out.release()
    cv.destroyAllWindows()

    # Define the Firebase bucket
    bucket = admin_storage.bucket()

    # Create a blob and upload the BytesIO object
    blob = bucket.blob('output_video_4.avi')

    with open(temp_path_out, 'rb') as tf:
        blob.upload_from_file(tf, content_type='video/avi')

    # blob.upload_from_file(video_bytes, content_type='video/mp4')

    # Make the blob publicly accessible and get the public URL
    blob.make_public()
    public_url = blob.public_url

    # processed_video.write(out)
    # processed_video.seek(0)

    # Delete the temporary file
    os.unlink(temp_path_out)

    return public_url


if __name__ == '__main__':
    app.run(debug=True)
