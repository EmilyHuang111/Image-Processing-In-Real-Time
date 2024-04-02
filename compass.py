import cv2 as cv
import numpy as np

#https://medium.com/@mithi/advanced-lane-finding-using-computer-vision-techniques-7f3230b6c6f2 line 32
#https://github.com/abhijitmahalle/lane-detection/tree/master 
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10302685/ example 1
#https://medium.com/lifeandtech/lane-detection-with-turn-prediction-1773c4819541 lines 10-15

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
