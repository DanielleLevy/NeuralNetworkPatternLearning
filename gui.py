from tkinter import Tk, Button, Label, Entry
from tkinter import ttk
import buildnet
import runnet

root = Tk()
root.geometry("400x300")  # Set the size of the window
root.title("Neural Network Pattern Learning")  # Set the title of the window

# Header
header_label = Label(root, text="Choose an Option:", font=("Arial", 16, "bold"))
header_label.pack(pady=10)

# Create labels and entry fields for buildnet
test_label = Label(root, text="Test File:")
test_label.pack()
test_entry = Entry(root)
test_entry.pack()

train_label = Label(root, text="Training File:")
train_label.pack()
train_entry = Entry(root)
train_entry.pack()

buildnet_frame = ttk.Frame(root)
buildnet_frame.pack()

def run_buildnet():
    buildnet.run_buildnet(test_entry.get(), train_entry.get())
    root.destroy()  # Destroy the GUI window after button press

buildnet_button = ttk.Button(buildnet_frame, text="Run buildnet", command=run_buildnet)
buildnet_button.pack(pady=10)

# Create labels and entry fields for runnet
structure_label = Label(root, text="Structure File:")
structure_label.pack()
structure_entry = Entry(root)
structure_entry.pack()

data_label = Label(root, text="Data File:")
data_label.pack()
data_entry = Entry(root)
data_entry.pack()

runnet_frame = ttk.Frame(root)
runnet_frame.pack()

def run_runnet():
    runnet.run_runnet(structure_entry.get(), data_entry.get())
    root.destroy()  # Destroy the GUI window after button press

runnet_button = ttk.Button(runnet_frame, text="Run runnet", command=run_runnet)
runnet_button.pack(pady=10)

root.mainloop()
