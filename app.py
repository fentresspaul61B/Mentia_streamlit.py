
import time
from IPython.display import Audio
from ipywebrtc import CameraStream, AudioRecorder, AudioStream

# actually I found this hack in some js code
# just pass mime type = )
camera = CameraStream(constraints = {
      'audio': True,
      'video': False
   },
   mimeType = 'audio/wav')
recorder = AudioRecorder(stream=camera)

# turn the recorder on
# still a bit rusty on whether I need to show it
# in a separate cell later to make it work
recorder.recording = True
# say something
print("Recording")
time.sleep(5)
print("Complete")
# turn the recorder off
recorder.recording = False
recorder.save()
# enjoy your wav(a typical user will be happy with compressed sound ofc)
