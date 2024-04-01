#import necessary libraries for image processing, log, or database
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

# Initialize webcam with a video path
video_file_path = "./video/IMG_0405.mp4"

webcam = cv.VideoCapture(video_file_path)

webcam_lock = threading.Lock()

# Update the label for log file

def update_label(image, label):
    # Create a PhotoImage object from the given image
    photo = ImageTk.PhotoImage(image=image)
    # Configure the label to display the given photo
    label.config(image=photo)
    # Store the reference to the PhotoImage object within the label to prevent it from being garbage collected
    label.image = photo

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

# Stop signal for video replay
stop_video_replay = threading.Event()

# List to store curvature values
curvature_values = []

# Number of values to use for curvature averaging
n_values_for_average = 10

# Function to undistort images
def undistort_img():
    
    # Generate object points for chessboard corners
    obj_pts = np.zeros((6*9, 3), np.float32)
    obj_pts[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Lists to store object points and image points
    objpoints = []
    imgpoints = []

    # Load calibration images
    images = glob.glob('camera_cal/*.jpg')
    img_size = None

    # Loop through images for calibration
    for indx, fname in enumerate(images):
        img = cv.imread(fname)
        if img is None:
            continue
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])

        # Convert image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

        # If corners found, add them to the lists
        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)

    # Raise exception if no valid images are found
    if img_size is None:
        raise Exception("No valid images found in 'camera_cal/*.jpg'.")

    # Calibrate camera using found points
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Undistort an example image
    dst = cv.undistort(img, mtx, dist, None, mtx)

    # Save calibration parameters
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump(dist_pickle, open('camera_cal/cal_pickle.p', 'wb'))

# Function to undistort image using calibration parameters
def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    # Load calibration parameters
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    
    # Undistort the image
    dst = cv.undistort(img, mtx, dist, None, mtx)
    
    return dst


def pipeline(img, s_thresh=(22, 255), sx_thresh=(15, 255)):
    # Undistort the input image
    img = undistort(img)
    
    # Make a copy of the undistorted image
    img = np.copy(img)
    
    # Convert the image to HLS color space
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS).astype(np.float64)
    l_channel = hls[:,:,1]  # Extract the L channel
    s_channel = hls[:,:,2]  # Extract the S channel
    h_channel = hls[:,:,0]  # Extract the H channel
    
    # Apply Sobel operator in x direction on L channel
    sobelx = cv.Sobel(l_channel, cv.CV_64F, 1, 1)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Apply thresholding to the Sobel x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Apply thresholding to the S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Create a color binary image combining the thresholded S channel and Sobel x gradient
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    # Combine binary images
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def perspective_warp(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    # Get the size of the input image
    img_size = np.float32([(img.shape[1],img.shape[0])])
    
    # Scale the source and destination points by the size of the image
    src = src * img_size
    dst = dst * np.float32(dst_size)
    
    # Compute the perspective transformation matrix
    M = cv.getPerspectiveTransform(src, dst)
    
    # Apply perspective transform to the image
    warped = cv.warpPerspective(img, M, dst_size)
    
    return warped


def inv_perspective_warp(img, 
                         dst_size=(1280,720),
                         src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                         dst=np.float32([(0.43,0.85),(0.58,0.85),(0.1,1),(1,1)])):
    # Get the size of the input image
    img_size = np.float32([(img.shape[1],img.shape[0])])
    
    # Scale the source and destination points by the size of the image
    src = src * img_size
    dst = dst * np.float32(dst_size)
    
    # Compute the perspective transformation matrix
    M = cv.getPerspectiveTransform(src, dst)
    
    # Apply inverse perspective warp to the image
    warped = cv.warpPerspective(img, M, dst_size)

    return warped

# Function to calculate histogram of an image
def get_hist(img):
    # Sum pixel values along vertical axis
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

# Initialize coefficients for left and right lanes
left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

# Function to perform sliding window search for lane lines
def sliding_window(img, nwindows=9, margin=150, minpix=1, draw_windows=True):
    global left_a, left_b, left_c, right_a, right_b, right_c 
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255

    # Calculate histogram of bottom half of the image
    histogram = get_hist(img)
    
    # Find the peak in the left and right halves of the histogram
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Calculate the height of each sliding window
    window_height = int(img.shape[0] / nwindows)

    # Find the indices of all non-zero elements in the binary image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])  # y-coordinates of non-zero elements
    nonzerox = np.array(nonzero[1])  # x-coordinates of non-zero elements

    # Initialize the starting x-coordinate for the left and right lanes
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Lists to store indices of non-zero elements belonging to the left and right lanes
    left_lane_inds = []
    right_lane_inds = []


   
    # Iterate through each window for lane detection
    for window in range(nwindows):
       
        # Define the y-coordinate range for the current window
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        
        # Define the x-coordinate range for the left lane within the current window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        
        # Define the x-coordinate range for the right lane within the current window
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

      
        # Check if drawing of window rectangles is enabled
        if draw_windows == True:
           # Draw a rectangle representing the search window for the left lane on the output image
           cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
           (100,255,255), 3) 
           # Draw a rectangle representing the search window for the right lane on the output image
           cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
           (100,255,255), 3) 

        
        # Determine the indices of non-zero pixels within the search windows for the left and right lanes
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append the indices of non-zero pixels within the left and right lane search windows to their respective lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If there are enough good left lane indices (pixels), update the current x-position for the left lane
        if len(good_left_inds) > minpix:
         leftx_current = int(np.mean(nonzerox[good_left_inds]))

        # If there are enough good right lane indices (pixels), update the current x-position for the right lane
        if len(good_right_inds) > minpix:        
         rightx_current = int(np.mean(nonzerox[good_right_inds]))

        
    
        # Concatenate the lists of indices to obtain a single array of indices for left and right lanes
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract x and y coordinates of the left and right lane pixels using the indices
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second-degree polynomial to the left and right lane pixels
    if lefty.size > 0 and leftx.size > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])
        
        # Update the moving average of the left lane coefficients
        left_fit_[0] = np.mean(left_a[-10:])
        left_fit_[1] = np.mean(left_b[-10:])
        left_fit_[2] = np.mean(left_c[-10:])
          
    else:
        # If no left lane pixels are detected, print a message
        print("No left lane pixels detected.")

    # Fit a second-degree polynomial to the right lane pixels
    if len(righty) > 0 and len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)

        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])
        
        # Update the moving average of the right lane coefficients
        right_fit_[0] = np.mean(right_a[-10:])
        right_fit_[1] = np.mean(right_b[-10:])
        right_fit_[2] = np.mean(right_c[-10:])
        
    else:
        # If no right lane pixels are detected, print a message
        print("No left lane pixels detected.")

    # Generate y values for plotting the fitted polynomial curves
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )

    # Calculate x values using the fitted polynomial curves for left and right lanes
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    # Color the detected lane pixels on the output image
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    # Return the output image, fitted polynomial curves, coefficients, and y values for plotting
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty


def get_curve(img, leftx, rightx, ploty):
    # Define the maximum y-value for evaluation
    y_eval = np.max(ploty)
    
    # Define the length of the lane in meters and the width of the lane in meters
    lane_length_m = 30.5  
    lane_width_m = 3.7    
    
    # Calculate the conversion factors from pixels to meters
    ym_per_pix = lane_length_m / img.shape[0]  
    xm_per_pix = lane_width_m / img.shape[1]   

    # Fit second-degree polynomials to the left and right lane pixels in meters
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Define a small epsilon value to avoid division by zero
    epsilon = 1e-6

    # Calculate the radius of curvature for the left and right lanes in meters
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / (np.absolute(2 * left_fit_cr[0])+epsilon)
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / (np.absolute(2 * right_fit_cr[0])+epsilon)

    # Determine the direction of the curvature (positive for left, negative for right)
    left_curve_dir = np.sign(2 * left_fit_cr[0])
    right_curve_dir = np.sign(2 * right_fit_cr[0])

    # Adjust the curvature values based on their direction
    left_curverad *= left_curve_dir
    right_curverad *= right_curve_dir

    # Calculate the position of the car relative to the lane center in meters
    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center = (car_pos - lane_center_position) * xm_per_pix / 10  

    # Return the left and right lane curvatures and the car position
    return (left_curverad, right_curverad, center)


def draw_lanes(img, left_fit, right_fit, ploty):
    # Create an empty color image with the same dimensions as the input image
    color_img = np.zeros_like(img)
    
    # Generate arrays representing the left and right lane lines
    left_lane = np.array([np.transpose(np.vstack([left_fit, ploty]))], dtype=np.int32)
    right_lane = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))], dtype=np.int32)

    # Draw the left lane line on the color image
    for i in range(len(ploty)-1):
        cv.line(color_img, (int(left_lane[0][i][0]), int(left_lane[0][i][1])), 
                (int(left_lane[0][i+1][0]), int(left_lane[0][i+1][1])), (0, 255, 0), thickness=30)

    # Draw the right lane line on the color image
    for i in range(len(ploty)-1):
        cv.line(color_img, (int(right_lane[0][i][0]), int(right_lane[0][i][1])), 
                (int(right_lane[0][i+1][0]), int(right_lane[0][i+1][1])), (0, 255, 0), thickness=30)

    # Calculate the center lane line and draw it on the color image
    center_fit = (left_fit + right_fit) / 2
    for i in range(len(ploty)-1):
        cv.line(color_img, (int(center_fit[i]), int(ploty[i])), (int(center_fit[i+1]), int(ploty[i+1])), (255, 0, 0), thickness=30)
    
    # Apply inverse perspective transform to get a bird's eye view
    inv_perspective = inv_perspective_warp(color_img, 
                                           dst_size=(img.shape[1], img.shape[0]),
                                           src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),  # Update these points
                                           dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]))  # Update these points

    # Combine the original image with the warped image
    result = cv.addWeighted(img, 1, inv_perspective, 1., 0)
    
    # Return the resulting image
    return result

def calculate_angle(curvature):
    # Define minimum and maximum curvature values
    curvature_min = 200
    curvature_max = 2000

    # Check if the absolute value of curvature is within the specified range
    if abs(curvature) <= curvature_min:
        angle = 45  # Set angle to 45 degrees
    elif abs(curvature) >= curvature_max:
        angle = 0  # Set angle to 0 degrees
    else:
        # Calculate angle based on the curvature within the specified range
        angle = (1 - (abs(curvature) - curvature_min) / (curvature_max - curvature_min)) * 45
    
    # Adjust the angle based on the sign of curvature
    angle *= np.sign(curvature)
    
    return angle


def draw_compass(img, curvature, position=None):
    # Set default position for the compass
    if position is None:
        position = (img.shape[1] // 2, 50)

    # Define parameters for drawing the compass
    compass_radius = 100
    line_thickness = 4
    compass_color = (255, 255, 255)  # White color for compass
    arm_color = (0, 0, 255)  # Red color for compass arm

    # Calculate the angle using the curvature value
    angle = calculate_angle(curvature)
    
    # Calculate the end point of the compass arm
    end_x = int(position[0] + compass_radius * np.sin(np.radians(angle)))
    end_y = int(position[1] - compass_radius * np.cos(np.radians(angle)))

    # Draw the ellipse representing the compass
    cv.ellipse(img, position, (compass_radius, compass_radius), 180, 0, 180, compass_color, thickness=line_thickness)
    
    # Draw the horizontal line of the compass
    cv.line(img, (position[0] - compass_radius, position[1]), (position[0] + compass_radius, position[1]), compass_color, line_thickness)
  
    # Draw the arm of the compass indicating the direction
    cv.line(img, position, (end_x, end_y), arm_color, line_thickness)

    # Return the image with the compass drawn on it
    return img

def vid_pipeline(img):
    global running_avg
    global index
    global curvature_values, n_values_for_average
    
    # Process the image using the pipeline function
    img_ = pipeline(img)
    # Perform perspective warp transformation
    img_ = perspective_warp(img_)
    # Apply sliding window method to detect lanes and obtain curve information
    out_img, curves, lanes, ploty = sliding_window(img_, draw_windows=False)
    # Calculate curvature of the detected lanes
    curverad = get_curve(img, curves[0], curves[1], ploty)
    # Compute average lane curvature
    lane_curve = np.mean([curverad[0], curverad[1]])
    # Convert lane curvature from meters to feet
    lane_curve_ft = lane_curve * 3.28084 
    # Append the current lane curvature to the curvature history
    curvature_values.append(lane_curve_ft)
    # Keep only the last n_values_for_average curvature values for averaging
    if len(curvature_values) > n_values_for_average:
        curvature_values.pop(0) 
    
    # Calculate the average curvature from the curvature history
    avg_curvature = sum(curvature_values) / len(curvature_values)
    # Draw lanes on the image
    img = draw_lanes(img, curves[0], curves[1], ploty)
    
    # Set position for compass display
    compassPosition = (img.shape[1] // 2, 700) 
    # Draw compass indicating the average curvature direction
    img = draw_compass(img, avg_curvature, compassPosition)

    return img

def safe_webcam_read(cap, max_attempts=5):
    attempts = 0
    # Try reading from the webcam for a maximum number of attempts
    while attempts < max_attempts:
        ret, frame = cap.read()
        # If successfully read, return the frame
        if ret:
            return True, frame
        # If reading fails, print an error message and wait before retrying
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts} failed to read from webcam.")
        time.sleep(0.1)  
    # Return False and None if reading fails after max_attempts
    return False, None  

def load_video_raw(currentState, userName, frame, stop_event):
    global webcam, webcam_lock
    currentState.config(text="Current State: Camera loaded")
    # Create a label for displaying the raw video feed
    video_label2 = tk.Label(frame)
    video_label2.grid(row=2, column=0, columnspan=2)
    
    if not webcam.isOpened():
        currentState.config(text="Current State: No Connection")
        return  
    
    frame_rate = 16  # Frame rate for video display
    # Continuously read frames from the webcam until stop event is set
    while not stop_event.is_set():
        with webcam_lock:
            # Safely read frame from the webcam
            ret, frame = safe_webcam_read(webcam)
        if ret:
            # Convert the frame to RGB format and resize it
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            rgb_frame = cv.resize(rgb_frame, (256, 256))
            image = Image.fromarray(rgb_frame)
            # Convert the RGB image to a Tkinter-compatible format
            photo = ImageTk.PhotoImage(image=image)
            # Update the video label with the new frame
            video_label2.config(image=photo)
            video_label2.image = photo  
            # Schedule the next update after the specified time interval
            root.after(0, update_label, Image.fromarray(rgb_frame), video_label2)
            # Control frame rate by waiting for a short duration
            time.sleep(1 / frame_rate)  
        else:
            # If reading fails, reset the webcam to the beginning
            with webcam_lock:
                webcam.set(cv.CAP_PROP_POS_FRAMES, 0)

def load_video_processed(currentState, userName, frame, stop_event):
    global webcam, webcam_lock
    currentState.config(text="Current State: Overlay Loaded")
    # Create a label for displaying the processed video feed
    video_label1 = tk.Label(frame)
    video_label1.grid(row=2, column=0, columnspan=2)
    
    frame_rate = 16  # Frame rate for video display

    if not webcam.isOpened():
        currentState.config(text="Current State: No Connection")
        return
    else:
        # Continuously read frames from the webcam until stop event is set
        while not stop_event.is_set():
            with webcam_lock:
                # Safely read frame from the webcam
                ret, original_frame = safe_webcam_read(webcam)
                # If reading fails, reset the webcam to the beginning and continue
                if not ret:
                    webcam.set(cv.CAP_PROP_POS_FRAMES, 0)
                    continue  
            # Process the original frame using the video pipeline
            processed_frame = vid_pipeline(original_frame)
            # Convert the processed frame to RGB format and resize it
            rgb_frame = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
            rgb_frame = cv.resize(rgb_frame, (256, 256))
            image = Image.fromarray(rgb_frame)
            # Convert the RGB image to a Tkinter-compatible format
            photo = ImageTk.PhotoImage(image=image)
            # Update the video label with the processed frame
            video_label1.config(image=photo)
            video_label1.image = photo
            # Schedule the next update after the specified time interval
            root.after(0, update_label, Image.fromarray(rgb_frame), video_label1)
            # Control frame rate by waiting for a short duration
            time.sleep(1 / frame_rate)

