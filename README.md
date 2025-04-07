# BigSister
Big Sister is always watching

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
# You might need to install system libraries for PyAudio and webrtcvad depending on your OS.
# For example, on Debian/Ubuntu:
# sudo apt-get update
# sudo apt-get install portaudio19-dev python3-pyaudio
```

2. Set your Google API key and your Neuphonic API key as an environment variable:
```
export GOOGLE_API_KEY="google-api-key"
export NEUPHONIC_API_KEY="neuphonic-api-key"
```

3. Run the application:
```
python camera.py
```

## Usage
- The application will open your camera and start recording audio in 5-second segments.
- Video frames and audio will be processed through the Gemini 2.0 Flash API.
- Text analysis results will be displayed in the terminal.
- Press 'q' in the camera window to quit.
- Perform actions like smoking, drinking water, using your phone on camera.
- Your Big Sister credits will increase or decrease depending on your behaviour.
- Big Sister will tell you to either stop or continue your behaviour.

## Customisation
- You can define your own behaviors for Big Sister to detect at line 132 by modifying the prompt.
- Then, at line 174, add the appropriate response Big Sister should say to the user when those behaviors are detected, and adjust their social credit score accordingly.



## Note

The application creates temporary files for processing that are automatically cleaned up when the program exits. 