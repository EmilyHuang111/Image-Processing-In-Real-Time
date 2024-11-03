# Robot Control GUI Application

This project is a Python GUI application for controlling a robot, logging user activity, and displaying live and processed video feeds. Built using Tkinter, OpenCV, and Flask, the application allows users to control a robot's movement, view real-time video, and log activity through a secure login system.

## Features

- **User Authentication**: Login and registration system with SQLite, using salted password hashing with `pbkdf2_hmac`.
- **Robot Control**: Control robot directions (Forward, Backward, Left, Right, Start, Stop) through HTTP requests to a Flask server.
- **Video Feeds**: Live video feed and processed video display with lane detection and curvature indicators.
- **Logging**: Logs all actions (login, logout, robot control) with timestamps, viewable in the GUI.
- **Lane Detection and Compass**: Lane detection pipeline with undistortion, perspective transformation, and curvature calculations. Displays compass to indicate lane curvature.

## Project Structure

- **login.py**: Contains functions for the user login process.
- **register.py**: Manages user registration, ensuring username uniqueness and password strength.
- **GUI.py**: Main GUI file, handling Tkinter interface for robot control, logging, and video feed.
- **database.db**: SQLite database file storing user information.

## Setup

### Prerequisites

- Python 3.x
- SQLite3
- Required Python packages:

  ```bash
  pip install tkinter opencv-python requests Pillow pytz
