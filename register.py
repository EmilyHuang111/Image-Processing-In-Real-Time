import os  #type: ignore
import tkinter as tk
from tkinter import ttk #Imported for updating GUI style
from tkinter import Toplevel, messagebox
from login import *
import sqlite3 #Datebase
import hashlib

#Function to create menu to register user for a new account
def register_menu():
  global firstNameEntry, lastNameEntry, usernameEntry, passwordEntry, register_window, register_warning
  #creates a register window with title register
  register_window = Toplevel(root)
  register_window.title("Register")
  register_window.geometry("500x240")

  #creates input lables and input fields
  firstNameLabel = ttk.Label(register_window, text="First Name*:")
  lastNameLabel = ttk.Label(register_window, text="Last Name*:")
  usernameLabel = ttk.Label(register_window, text="Username*:")
  passwordLabel = ttk.Label(register_window, text="Password*:")
  firstNameEntry = ttk.Entry(register_window)
  lastNameEntry = ttk.Entry(register_window)
  usernameEntry = ttk.Entry(register_window)
  passwordEntry = ttk.Entry(register_window, show="*")
  noteLabel = ttk.Label(register_window, text="         * Input Required")
  register_warning = tk.Label(register_window, text="",fg='#f00') #Warning for input information not meeting requirements

  #creates a submit button and save the register data to database
  saveButton = tk.Button(register_window, text="Submit")
  saveButton.configure(command=save_register)

  #Position buttons and input fields in grid layout
  firstNameLabel.grid(row=0, column=0, padx=30, pady=5)
  firstNameEntry.grid(row=0, column=1, padx=30, pady=5)
  lastNameLabel.grid(row=1, column=0, padx=30, pady=5)
  lastNameEntry.grid(row=1, column=1, padx=30, pady=5)
  usernameLabel.grid(row=2, column=0, padx=30, pady=5)
  usernameEntry.grid(row=2, column=1, padx=30, pady=5)
  passwordLabel.grid(row=3, column=0, padx=30, pady=5)
  passwordEntry.grid(row=3, column=1, padx=30, pady=5)
  noteLabel.grid(row=4, column=0, padx=10, pady=5)
  register_warning.grid(row=5, column=1, columnspan=2, padx=1, pady=5)
  saveButton.grid(row=6, column=1, columnspan=2, padx=10, pady=10)

# Save registration information based on register menu inputs
def save_register():
  global register_window,register_warning
  assert firstNameEntry is not None
  first_name = firstNameEntry.get()
  assert lastNameEntry is not None
  last_name = lastNameEntry.get()
  assert usernameEntry is not None
  username = usernameEntry.get()
  assert passwordEntry is not None
  password = passwordEntry.get()
  salt = os.urandom(16) 
  #Check if first name was entered 
  if len(first_name) < 1:
     register_warning.config(text= "First Name is required.")
  else:
    #Check if username exits in the database
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    cursor.execute(
        "SELECT first_name, password_hash, salt FROM user_information WHERE username = ?",
        (username, ))
    result = cursor.fetchone()
    connection.close()
    if result:
      register_warning.config(text= "Username exits in the database.")
    else:
      #Check password inputs to meet requirements for 8 charactors, a number, an upper case letter and a lowercase letter.
      if len(password) < 8:
         register_warning.config(text= "Password to be at least 8 characters.")
      elif not any(chr.isdigit() for chr in password):
         register_warning.config(text= "Password needs a number.")
      elif not any(chr.isupper() for chr in password):
         register_warning.config(text= "Password needs an uppercase letter.")     
      elif not any(chr.islower() for chr in password):
         register_warning.config(text= "Password needs a lowercase letter.") 
      else:
         register_warning.config(text= "") 
         password_bytes = password.encode('utf-8')
         password_hash = hashlib.pbkdf2_hmac('sha256', password_bytes, salt, 100000)
         connection = sqlite3.connect("database.db")  
         cursor = connection.cursor()  
         #Save inputs to the database for first name, last name, username and password
         cursor.execute(
          "INSERT INTO user_information (first_name, last_name, username, password_hash, salt) VALUES (?, ?, ?, ?, ?)",
          (first_name, last_name, username, password_hash, salt))
         connection.commit()
         connection.close()
         #Show message that the user account has been created.
         messagebox.showinfo(
          "Register",
          f"User Account created successfully:\nFirst Name: {first_name}\nLast Name: {last_name}\nUsername: {username}"
         )  
         assert register_window is not None
         register_window.destroy()

