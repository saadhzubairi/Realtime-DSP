# device_info.py
# print audio device information

import pyaudio

p = pyaudio.PyAudio()

K = p.get_device_count()  # number of audio devices
print(K)

info = p.get_device_info_by_index(0)
type(info)

info['name']

info['maxInputChannels']

info['maxOutputChannels']

info['defaultSampleRate']



info = p.get_device_info_by_index(1)

print( p.get_device_info_by_index(1) )


for k in range(K):
	print(k)
	print( p.get_device_info_by_index(k) )

