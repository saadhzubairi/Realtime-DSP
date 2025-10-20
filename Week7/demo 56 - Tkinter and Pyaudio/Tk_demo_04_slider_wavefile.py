# Tk_demo_04_slider_wavefile.py
# TKinter demo
# Play a sinusoid using Pyaudio. Use two sliders to adjust the frequency and gain.
# Save output to wave file

from math import cos, pi 
import pyaudio, struct
import tkinter as Tk   	
import wave

RATE = 8000     # rate (samples/second)
gain = 0.2 * 2**15

# Create wave file
file_name = 'output.wav'          # Name of output wavefile
wf = wave.open(file_name, 'w')    # wf : wave file
wf.setnchannels(1)                # one channel (mono)
wf.setsampwidth(2)                # two bytes per sample (16 bits per sample)
wf.setframerate(RATE)               # samples per second

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
f1.set(200)   # f1 : frequency of sinusoid (Hz)
gain.set(0.2 * 2**15)

# Define widgets
S_freq = Tk.Scale(root, label = 'Frequency', variable = f1, from_ = 100, to = 400, tickinterval = 100)
S_gain = Tk.Scale(root, label = 'Gain', variable = gain, from_ = 0, to = 2**15-1)
B_quit = Tk.Button(root, text = 'Quit', command = fun_quit)

# Place widgets
B_quit.pack(side = Tk.BOTTOM, fill = Tk.X)
S_freq.pack(side = Tk.LEFT)
S_gain.pack(side = Tk.LEFT)

BLOCKLEN = 256

# Create Pyaudio object
p = pyaudio.PyAudio()
stream = p.open(
  format = pyaudio.paInt16,  
  channels = 1, 
  rate = RATE,
  input = False, 
  output = True,
  frames_per_buffer = BLOCKLEN)            
  # specify low frames_per_buffer to reduce latency

output_block = [0] * BLOCKLEN
theta = 0
CONTINUE = True

print('* Start')
while CONTINUE:
  root.update()
  om1 = 2.0 * pi * f1.get() / RATE
  A = gain.get()
  for i in range(0, BLOCKLEN):
    output_block[i] = int( A * cos(theta) )
    theta = theta + om1
  if theta > pi:
  	theta = theta - 2.0 * pi
  binary_data = struct.pack('h' * BLOCKLEN, *output_block)   # 'h' for 16 bits
  stream.write(binary_data)

  wf.writeframes(binary_data)   # write signal to wave file

print('* Finished')

stream.stop_stream()
stream.close()
p.terminate()
