import numpy as np
import cv2 as cv
import argparse
parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
cap = cv.VideoCapture(args.image)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=10,
                      qualityLevel=0.5,
                      minDistance=5,
                      blockSize=15)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)


out_w = 300
out_h = 300
out = cv.VideoWriter('resources/output/output_video.mp4',
                     cv.VideoWriter_fourcc(*'mp4v'), 30, (out_w, out_h))


while (1):
    ret, frame = cap.read()
    if not ret:
        print('Video not giving frames')
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

    img2 = cv.rectangle(frame, (int(a) - 50, int(b) - 50),
                        (int(a) + 50, int(b)+50), 255, 2)
    cv.imshow('img2', img2)

    # EXTRACTING WINDOW OF SURFER
    out_window = cv.getRectSubPix(
        frame, (out_w, out_h), (int(a) - 50, int(b) - 50))

    # sharpen
    # sharpen_kernel = np.array([[-0.8, -1, -0.8],
    #                            [-1,  9, -1],
    #                            [-0.8, -1, -0.8]])
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    out_window = cv.filter2D(out_window, -1, sharpen_kernel)

    # filter
    # out_window = cv.medianBlur(out_window, 11)

    # show and write
    cv.imshow('zoomed-in window', out_window)
    out.write(out_window)

    # Unsharp Masking (USM): This method involves creating a blurred version of the image and subtracting it from the original image to create a sharpened version. It can be effective for sharpening small areas in an image.
    # High Pass Filtering: This method involves removing the low-frequency components of an image and retaining the high-frequency components, which can enhance details and edges. This can be effective for sharpening small areas in an image.
    # Local Contrast Enhancement: This method involves increasing the contrast in small areas of an image, which can make edges and details more visible. This can be done using tools like the "Clarity" slider in Adobe Lightroom.
    # Selective Sharpening: This method involves using a brush or selection tool to apply sharpening only to specific areas of an image that need it. This can be effective for sharpening small areas in an image while minimizing the impact on other parts of the image.

    # TRYING BasicVSR++

    # key = cv.waitKey(0) & 0xFF
    # if key == ord('n'):
    #     continue
    k = cv.waitKey(30) & 0xff
    if k == ord('q'):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
