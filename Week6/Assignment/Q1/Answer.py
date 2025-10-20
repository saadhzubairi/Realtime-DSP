import tkinter as Tk

def update_label():
    name = entry.get()
    if sound_var.get():
        result.set(f"Hi {name}, volume set to {scale.get()}")
    else:
        result.set(f"Hi {name}, mute mode. Volume ignored.")

root = Tk.Tk()
root.title("Sample Tkinter GUI")
root.geometry("300x350")

# Tk variables
result = Tk.StringVar()
sound_var = Tk.BooleanVar(value=True)

# Widgets
label_title = Tk.Label(root, text="Welcome to My GUI", font=("Arial", 14, "bold"))
label_name = Tk.Label(root, text="Enter your name:")
entry = Tk.Entry(root)
scale = Tk.Scale(root, from_=0, to=100, orient=Tk.HORIZONTAL, label="Volume")
check_sound = Tk.Checkbutton(root, text="Enable sound", variable=sound_var)
button = Tk.Button(root, text="Submit", command=update_label)
label_result = Tk.Label(root, textvariable=result, fg="blue")

# Layout
label_title.pack(pady=5)
label_name.pack()
entry.pack()
scale.pack()
check_sound.pack()
button.pack(pady=5)
label_result.pack(pady=10)
Tk.Button(root, text="Quit", command=root.quit, background="#DEA4A4", width=5).pack(fill=Tk.X)

root.mainloop()
