import os  # Import os module for system operations
import sqlite3  # Import sqlite3 module for database operations
import hashlib  # Import hashlib module for cryptographic hashing
import tkinter as tk  # Import tkinter module for GUI
from tkinter import Toplevel, messagebox  # Import specific classes from tkinter
from tkinter import ttk  # Import ttk module for themed widgets
import requests  # Import requests module for HTTP requests
from login import *  # Import login functions from login module
from register import *  # Import register functions from register module

# Create database instance
connection = sqlite3.connect("database.db")  # Connect to SQLite database file named "database.db"
cursor = connection.cursor()  # Create a cursor object to execute SQL commands

# Create tables in the database for user information
cursor.execute('''CREATE TABLE IF NOT EXISTS user_information (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  first_name TEXT,
                  last_name TEXT,
                  username TEXT,
                  password_hash BLOB,
                  salt BLOB)''')
connection.commit()  # Commit changes to the database
connection.close()  # Close database connection

# Function to exit the program
def exit_app(): 
    root.quit()

# Create starting login/register window 
# Create a width for all the buttons
button_width = 10
frm = ttk.Frame(root, padding=10)  # Create a frame with padding
frm.grid()  # Grid layout manager for the frame
ttk.Label(frm, text="Login with an existing account:").grid(row=0,column=0,padx=20,pady=10)  # Label for login
login_button = ttk.Button(frm, text="Login", command=login_menu, width=button_width)  # Login button
login_button.grid(row=0, column=1, padx=20, pady=10)  # Grid layout for login button
ttk.Label(frm, text="Or register a new account:").grid(row=1,column=0,padx=20,pady=10)  # Label for registration
register_button = ttk.Button(frm, text="Register", command=register_menu, width=button_width)  # Register button
register_button.grid(row=1, column=1, padx=20, pady=10)  # Grid layout for register button
exit_button = ttk.Button(frm, text="Exit", command=exit_app)  # Exit button
exit_button.grid(row=2, column=0, columnspan=2, padx=20, pady=10)  # Grid layout for exit button

root.mainloop()  # Run the Tkinter event loop



#https://www.youtube.com/watch?v=iRTuCYx6quQ
#https://www.geeksforgeeks.org/perspective-transformation-python-opencv/
#https://www.hackster.io/kemfic/curved-lane-detection-34f771
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
# https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/
# https://www.youtube.com/watch?v=CvJN_jSVm30
# https://www.youtube.com/watch?v=eLTLtUVuuy4


