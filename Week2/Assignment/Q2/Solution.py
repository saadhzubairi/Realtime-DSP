from math import cos, pi
import pyaudio, struct

Fs = 8000
T = 1
N = T * Fs

r = 0.999
omega = 2*pi*400/Fs      # ~400 Hz tone
a1 = -2 * r * cos(omega)
a2 = r * r

# h[n] = r^n cos(omega n) u[n]
b0 = 1.0
b1 = a1 / 2.0    # = -r cos(omega)
b2 = 0.0

# maintaing states
x1 = x2 = 0.0
y1 = y2 = 0.0

gain = 10000.0

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=Fs, input=False, output=True)

for n in range(N):
    # Use impulse as input signal
    if n == 0:
        x0 = 1.0
    else:
        x0 = 0.0


    y0 = (b0*x0 + b1*x1 + b2*x2) - a1*y1 - a2*y2
    
    # Delays
    x2, x1 = x1, x0
    y2, y1 = y1, y0

    maxGain = ((2**15)/abs(y0)).__floor__() - 1
    if(gain > maxGain):
        print(y0,'/tmax gain:',maxGain)
        gain = maxGain
        
    output_value = gain * y0
    output_string = struct.pack('h', int(output_value))   # 'h' for 16 bits
    stream.write(output_string)

print("* Finished *")
stream.stop_stream(); stream.close(); p.terminate()
