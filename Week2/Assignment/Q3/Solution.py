import pyaudio
import struct

Fs = 8000
T = 2
N = T * Fs

a1_L = -0.5
a2_L = 0.8

a1_R = -1.9
a2_R = 0.998

y1_L = 0.0
y2_L = 0.0
y1_R = 0.0
y2_R = 0.0

gain = 5000.0

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=2,
                rate=Fs,
                input=False,
                output=True)

for n in range(0, N):
    if n == 0:
        x0 = 1.0
    else:
        x0 = 0.0

    y0_L = x0 - a1_L * y1_L - a2_L * y2_L
    y0_R = x0 - a1_R * y1_R - a2_R * y2_R
    
    y2_L, y1_L = y1_L, y0_L
    y2_R, y1_R = y1_R, y0_R

    output_value_L = gain * y0_L
    output_value_R = gain * y0_R
    output_string = struct.pack('<hh', int(output_value_L), int(output_value_R))
    stream.write(output_string)

print("* Finished *")

stream.stop_stream()
stream.close()
p.terminate()
