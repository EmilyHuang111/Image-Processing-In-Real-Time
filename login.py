import os  # Import os module for system operations (ignoring type hints)
import tkinter as tk  # Import tkinter module for GUI
from tkinter import ttk  # Imported for updating GUI style
from tkinter import Toplevel, messagebox  # Import specific classes from tkinter
import sqlite3  # Import sqlite3 module for database operations
import hashlib  # Import hashlib module for cryptographic hashing
from GUI import *  # Import GUI functions from GUI module

# Create login menu
def login_menu():
    global loginUsernameEntry, loginPasswordEntry, login_window, login_warning
    global first_name, last_name
    login_window = tk.Toplevel()  # Create a new top-level window for login
    login_window.title("Login")  # Set window title
    login_window.geometry("500x160")  # Set window size

    # Create labels and entry fields for username and password
    loginLabel = ttk.Label(login_window, text="Username:")
    loginUsernameEntry = ttk.Entry(login_window)
    loginLabel.grid(row=0, column=0, padx=30, pady=5)
    loginUsernameEntry.grid(row=0, column=1, padx=30, pady=5)
    loginPasswordLabel = ttk.Label(login_window, text="Password:")
    loginPasswordEntry = ttk.Entry(login_window, show="*")
    loginPasswordLabel.grid(row=1, column=0, padx=30, pady=5)
    loginPasswordEntry.grid(row=1, column=1, padx=30, pady=5)
    login_warning = tk.Label(login_window, text="", fg='#f00')  # Warning label for input validation
    login_warning.grid(row=2, column=1, columnspan=2, padx=30, pady=5)

    # Create login button
    loginButton = tk.Button(login_window, text="Login")
    loginButton.configure(command=login)
    loginButton.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    # Create text for retry or register
    loginNoteLabel = tk.Label(
        login_window,
        text="Failed to login, please try again or register for new account.")
    loginNoteLabel.grid(row=4, column=0, columnspan=2, padx=10, pady=20)

# Function to login user with the user inputs of username and password
def login():
    global welcome_window, login_window, first_name, login_warning, last_name, username
    assert loginUsernameEntry is not None
    enteredUsername = loginUsernameEntry.get()
    assert loginPasswordEntry is not None
    enteredPassword = loginPasswordEntry.get()

    # Verify from the database to see if username exists
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    cursor.execute(
        "SELECT first_name, last_name, password_hash, salt FROM user_information WHERE username = ?",
        (enteredUsername, ))
    result = cursor.fetchone()
    connection.close()
    if result:
        storedFirstName, storedLastName, storedPasswordHash, storedSalt = result
        enteredPasswordBytes = enteredPassword.encode('utf-8')
        enteredPasswordHash = hashlib.pbkdf2_hmac('sha256', enteredPasswordBytes,
                                                  storedSalt, 100000)
        # If the login user's password is the same as the password in the database
        # Show welcome message and GUI
        if storedPasswordHash == enteredPasswordHash:
            first_name = storedFirstName
            last_name = storedLastName
            loggedin_window = tk.Toplevel()
            loggedin_window.title("Logged In")
            loggedin_window.geometry("450x40")
            loggedin_label = tk.Label(loggedin_window, text=f"Welcome, {enteredUsername}!")
            loggedin_label.pack(padx=20, pady=10)
            activity = f"{enteredUsername} has logged in."
            log_activity(activity)
            # Show graphic user interface
            GUI(first_name, last_name, enteredUsername, loggedin_window)
            # Close out root and login windows
            assert login_window is not None
            login_window.destroy()
            root.withdraw()
        else:
            login_warning.config(text="Password is incorrect. Try again.")
    else:
        login_warning.config(text="Username not found. Try again or register.")
