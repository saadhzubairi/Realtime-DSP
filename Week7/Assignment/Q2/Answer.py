from math import cos, pi
import pyaudio, struct
import tkinter as Tk
import wave

RATE = 8000
gain = 0.2 * 2**15

# Create wave file
file_name = 'output.wav'
wf = wave.open(file_name, 'w')
wf.setnchannels(1)
wf.setsampwidth(2)
wf.setframerate(RATE)

def fun_quit():
  global CONTINUE
  print('Good bye')
  CONTINUE = False

# Define Tkinter root
root = Tk.Tk()

# Define Tk variables
f1 = Tk.DoubleVar()
gain = Tk.DoubleVar()

# Initialize Tk variables
f1.set(200)
gain.set(0.2 * 2**15)

# Define widgets
S_freq = Tk.Scale(root, label='Frequency', variable=f1, from_=100, to=400, tickinterval=100)
S_gain = Tk.Scale(root, label='Gain', variable=gain, from_=0, to=2**15-1)
B_quit = Tk.Button(root, text='Quit', command=fun_quit)

# Place widgets
B_quit.pack(side=Tk.BOTTOM, fill=Tk.X)
S_freq.pack(side=Tk.LEFT)
S_gain.pack(side=Tk.LEFT)

BLOCKLEN = 256

# Create Pyaudio object
p = pyaudio.PyAudio()
stream = p.open(
  format=pyaudio.paInt16,
  channels=1,
  rate=RATE,
  input=False,
  output=True,
  frames_per_buffer=BLOCKLEN)

output_block = [0] * BLOCKLEN
theta = 0
CONTINUE = True

# ---- Added: store previous gain for smooth transition ----
A_prev = gain.get()

print('* Start')
while CONTINUE:
  root.update()
  om1 = 2.0 * pi * f1.get() / RATE
  A_target = gain.get()

  # ---- Modified section: interpolate gain smoothly ----
  for i in range(0, BLOCKLEN):
    A = A_prev + (A_target - A_prev) * (i / BLOCKLEN)
    output_block[i] = int(A * cos(theta))
    theta = theta + om1
    if theta > pi:
      theta = theta - 2.0 * pi

  A_prev = A_target  # update previous gain for next block

  binary_data = struct.pack('h' * BLOCKLEN, *output_block)
  stream.write(binary_data)
  wf.writeframes(binary_data)

print('* Finished')

stream.stop_stream()
stream.close()
p.terminate()