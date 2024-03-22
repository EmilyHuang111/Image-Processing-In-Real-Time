#Import necessary libraries
import io
import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk   
from datetime import datetime
import logging
import pytz
import threading
import time 
import pickle
import glob

# Root Menu for user registration and login
root = tk.Tk()
root.title("User Login")

# Set up logging with US/Eastern timezone
eastern = pytz.timezone('US/Eastern')

#Set up global variables for video streaming control
stop_video_raw = threading.Event()
stop_video_processed = threading.Event()

# Initialize webcam
#video_file_path = "./video/IMG_0304.mp4"
#video_file_path = "./video/project_video.mp4"
video_file_path = "./video/challenge_video.mp4"


webcam = cv.VideoCapture(video_file_path)

webcam_lock = threading.Lock()

# Updat the lable for log file
def update_label(image, label):
    photo = ImageTk.PhotoImage(image=image)
    label.config(image=photo)
    label.image = photo  # Keep a reference

#Create a log file to track activities
# Set up logging
logging.basicConfig(
    filename='activity_log.txt',
    level=logging.INFO,
    format='%(message)s',
)

# Function for the log information including date and time and message
def log_activity(message):
  current_time = datetime.now(eastern).strftime('%Y-%m-%d %I:%M:%S %p')
  logging.info(f"{current_time} - {message}")

# Global variable to control video replay
stop_video_replay = threading.Event()

curvature_values = []  # Stores recent curvature values for moving average calculation
n_values_for_average = 10  # Number of values to calculate the moving average, adjust as needed

def undistort_img():
    # Prepare object points 0,0,0 ... 8,5,0
    obj_pts = np.zeros((6*9,3), np.float32)
    obj_pts[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    objpoints = []  # Stores all object points
    imgpoints = []  # Stores all image points

    # Get directory for all calibration images
    images = glob.glob('camera_cal/*.jpg')
    img_size = None

    for indx, fname in enumerate(images):
        img = cv.imread(fname)
        if img is None:
            continue  # Skip if image not loaded
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])  # Define img_size using the first successfully loaded image

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (9,6), None)

        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)

    if img_size is None:
        raise Exception("No valid images found in 'camera_cal/*.jpg'.")

    # Calibrate camera using the object points and image points
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv.undistort(img, mtx, dist, None, mtx)
    # Save camera calibration for later use
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump( dist_pickle, open('camera_cal/cal_pickle.p', 'wb') )

undistort_img()

def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    # Load camera calibration parameters (mtx: camera matrix, dist: distortion coefficients)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    
    # Undistort the input image using the camera matrix and distortion coefficients
    dst = cv.undistort(img, mtx, dist, None, mtx)
    
    # Return the undistorted image
    return dst


def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = undistort(img)
    img = np.copy(img)
    # Convert the undistorted image color space to Hue, Lighteness, and Saturation color space
    # Extract the L(lightness), S(Saturation), and H(hue) channels
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS).astype(np.float64)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]

    sobelx = cv.Sobel(l_channel, cv.CV_64F, 1, 1) # Take the derivative in x-direction
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    
    # Apply thresholding to the scaled Sobel gradient in the x-direction
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def perspective_warp(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    # Calculate the size of the input image
    img_size = np.float32([(img.shape[1],img.shape[0])])
    # Scale the source points by the image size
    src = src * img_size
    
    # Scale the destination points by the desired output size
    dst = dst * np.float32(dst_size)
    
    # Calculate the perspective transform matrix using source and destination points
    M = cv.getPerspectiveTransform(src, dst)
    
    # Apply the perspective transformation to warp the image
    warped = cv.warpPerspective(img, M, dst_size)
    
    # Return the warped image
    return warped


def inv_perspective_warp(img, 
                         dst_size=(1280,720),
                         src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                         dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    # Get the size of the input image
    img_size = np.float32([(img.shape[1],img.shape[0])])
    
    # Scale the source points to match the image size
    src = src * img_size
    
    # Scale the destination points to the desired output size
    dst = dst * np.float32(dst_size)
    
    # Given source and destination points, calculate the perspective transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    
    # Warp the image using OpenCV warpPerspective() to obtain the inverse perspective transform
    warped = cv.warpPerspective(img, M, dst_size)
    
    # Return the warped image
    return warped


def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows=9, margin=150, minpix = 1, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c 
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
    # Set height of windows
    window_height = int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
            cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/720  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    # Calculate the direction of the curve
    left_curve_dir = np.sign(2 * left_fit_cr[0])
    right_curve_dir = np.sign(2 * right_fit_cr[0])

    # Apply the sign convention to the curvature values
    left_curverad *= left_curve_dir
    right_curverad *= right_curve_dir

    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters with the sign indicating the curve direction
    return (left_curverad, right_curverad, center)

def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    
    # Calculate points for the left and right lanes
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    
    # Fill the area between the lanes with green
    cv.fillPoly(color_img, np.int_([points]), (0, 255, 0))  # Green color
    
    # Calculate and draw the centerline with increased thickness
    center_fit = (left_fit + right_fit) / 2
    for i in range(len(ploty)-1):
        cv.line(color_img, (int(center_fit[i]), int(ploty[i])), (int(center_fit[i+1]), int(ploty[i+1])), (255, 0, 0), thickness=10)
    
    # Apply inverse perspective warp to get the result back to the original perspective
    inv_perspective = inv_perspective_warp(color_img)
    
    # Overlay the lane markings on the original image with 50% transparency
    result = cv.addWeighted(img, 1, inv_perspective, 0.25, 0)
    return result


def calculate_angle(curvature):
    # Define curvature thresholds
    curvature_min = 200
    curvature_max = 2000
    
    # Check if curvature is within the bounds for linear interpolation
    if abs(curvature) <= curvature_min:
        # Curvature is below the minimum threshold
        angle = 45
    elif abs(curvature) >= curvature_max:
        # Curvature is above the maximum threshold
        angle = 0
    else:
        # Linearly interpolate the angle based on the curvature
        angle = (1 - (abs(curvature) - curvature_min) / (curvature_max - curvature_min)) * 45
    
    # Adjust angle based on the sign of the curvature
    angle *= np.sign(curvature)
    
    return angle

def draw_compass(img, curvature, position=None):
    if position is None:
        # Positioning at the mid-top of the frame, slightly above for visibility
        position = (img.shape[1] // 2, 50)

    compass_radius = 100
    line_thickness = 4
    compass_color = (255, 255, 255)  # White color for half-circle and line
    arm_color = (0, 0, 255)  # Red color for the compass arm

    # Calculate the angle for the compass arm
    angle = calculate_angle(curvature)
    
    # Calculate the end point of the compass arm
    end_x = int(position[0] + compass_radius * np.sin(np.radians(angle)))
    end_y = int(position[1] - compass_radius * np.cos(np.radians(angle)))

    # Draw the half-circle and base line for the compass
    cv.ellipse(img, position, (compass_radius, compass_radius), 180, 0, 180, compass_color, thickness=line_thickness)
    cv.line(img, (position[0] - compass_radius, position[1]), (position[0] + compass_radius, position[1]), compass_color, line_thickness)

    # Draw the compass arm
    cv.line(img, position, (end_x, end_y), arm_color, line_thickness)

    return img


def vid_pipeline(img):
    global running_avg
    global index
    global curvature_values, n_values_for_average
    img_ = pipeline(img)
    img_ = perspective_warp(img_)
    out_img, curves, lanes, ploty = sliding_window(img_, draw_windows=False)
    curverad =get_curve(img, curves[0], curves[1])
    lane_curve = np.mean([curverad[0], curverad[1]])
    lane_curve_ft = lane_curve * 3.28084  # Convert from meters to feet
    # Update the list of recent curvature values
    curvature_values.append(lane_curve_ft)
    if len(curvature_values) > n_values_for_average:
        curvature_values.pop(0)  # Remove the oldest value to maintain the list size
    
    # Calculate the moving average of curvature
    avg_curvature = sum(curvature_values) / len(curvature_values)
    img = draw_lanes(img, curves[0], curves[1])
    
    font = cv.FONT_HERSHEY_SIMPLEX
    fontColor = (255, 255, 255)  # White color for better visibility
    fontSize = 1.5  # Larger font size
    fontThickness = 3  # Make the text bolder
    textPosition = (50, 50)  # Top left corner
    
    # Display the lane curvature text
    cv.putText(img, 'Lane Curvature: {:.0f} ft'.format(lane_curve_ft), textPosition, font, fontSize, fontColor, fontThickness)
    #cv.putText(img, 'Vehicle offset: {:.4f} m'.format(curverad[2]), (570, 650), font, fontSize, fontColor, 2)
    # Draw the compass based on lane curvature
    compassPosition = (img.shape[1] // 2, 700)  # Below the text for compass
    img = draw_compass(img, avg_curvature, compassPosition)

    return img

def safe_webcam_read(cap, max_attempts=5):
    attempts = 0
    while attempts < max_attempts:
        ret, frame = cap.read()
        if ret:
            return True, frame
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts} failed to read from webcam.")
        time.sleep(0.1)  # Brief pause before retrying
    return False, None  # Indicate failure after all attempts

def load_video_raw(currentState, userName, frame, stop_event):
    global webcam, webcam_lock
    currentState.config(text="Current State: Camera loaded")
    video_label2 = tk.Label(frame)
    video_label2.grid(row=2, column=0, columnspan=2)
    
    if not webcam.isOpened():
        currentState.config(text="Current State: No Connection")
        return  # Exit the function if the webcam is not opened
    
    frame_rate = 16  # Define the frame rate for half of the original speed
    while not stop_event.is_set():
        with webcam_lock:
            ret, frame = safe_webcam_read(webcam)
        if ret:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            rgb_frame = cv.resize(rgb_frame, (256, 256))
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=image)
            video_label2.config(image=photo)
            video_label2.image = photo  # Keep a reference
            root.after(0, update_label, Image.fromarray(rgb_frame), video_label2)
            time.sleep(1 / frame_rate)  # Delay frame loading
        else:
            # If the video ends (no frame returned), rewind it to the start
            with webcam_lock:
                webcam.set(cv.CAP_PROP_POS_FRAMES, 0)

def load_video_processed(currentState, userName, frame, stop_event):
    global webcam, webcam_lock
    currentState.config(text="Current State: Overlay Loaded")
    video_label1 = tk.Label(frame)
    video_label1.grid(row=2, column=0, columnspan=2)
    
    frame_rate = 16  # Ensure consistent frame rate

    if not webcam.isOpened():
        currentState.config(text="Current State: No Connection")
        return
    else:
        while not stop_event.is_set():
            with webcam_lock:
                ret, original_frame = safe_webcam_read(webcam)
                if not ret:
                    # If the video ends, rewind it to the start
                    webcam.set(cv.CAP_PROP_POS_FRAMES, 0)
                    continue  # Skip the current iteration and try reading again
            processed_frame = vid_pipeline(original_frame)
            rgb_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
            rgb_frame = cv.resize(rgb_frame, (256, 256))
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=image)
            video_label1.config(image=photo)
            video_label1.image = photo
            root.after(0, update_label, Image.fromarray(rgb_frame), video_label1)
            time.sleep(1 / frame_rate)  # Maintain frame rate
