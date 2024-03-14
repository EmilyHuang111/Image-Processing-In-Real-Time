import cv2
import numpy as np

# Turn on Laptop's webcam
cap = cv2.VideoCapture("video.MOV")
output = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*'MPEG'),
        30, (1080, 1920))

while (True):
    ret, frame = cap.read()
        if (ret):

            # adding filled rectangle on each frame
            cv2.rectangle(frame, (100, 150), (500, 600),
                          (0, 255, 0), -1)

            # writing the new frame in output
            output.write(frame)
            cv2.imshow("output", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break

    # Locate points of the documents
    # or object which you want to transform
    pts1 = np.float32([[0, 300], [540, 460],
                       [0, 400], [640, 640]])
    pts2 = np.float32([[0, 0], [500, 0],
                       [0, 640], [400, 940]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (500, 600))

    # Wrap the transformed image
    cv2.imshow('frame', frame)  # Initial Capture
    cv2.imshow('frame1', result)  # Transformed Capture

    if cv2.waitKey(24) == 27:
        break

cap.release()
cv2.destroyAllWindows()
