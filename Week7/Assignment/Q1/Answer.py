import numpy as np
import pyaudio
import struct

def karplus_strong(
    frequency=150.0, 
    Fs=8000, 
    K=0.998):
    
    duration=10*50/frequency
    #print(duration)
    print("* Playing")
    N = int(Fs / frequency)
    buffer = np.random.uniform(-1, 1, N)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=Fs, output=True)
    
    num_samples = int(duration * Fs)
    MAXVALUE = 2**15 - 1
    
    # circular buffer index
    idx = 0

    for i in range(num_samples):
        y = buffer[idx]
        next_idx = (idx + 1) % N
        avg = 0.5 * (buffer[idx] + buffer[next_idx])
        buffer[idx] = K * avg  # feedback
        idx = next_idx
        # convert to bytes and play
        sample = struct.pack('h', int(y * MAXVALUE))
        stream.write(sample)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("* Finished")

karplus_strong(110,K=0.998)

