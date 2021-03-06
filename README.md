# astromech-binary-droidspeak-ibm-watson
Windows based keyword spotting and binary 'droid speak' beep language translation for your Astromech droid powered by IBM Watson.

# WORK IN PROGRESS
I have a lot of clean up. This is not perfect. I want to add a better mechanism for managing droid profiles without code edits. Maybe command line switches for known droids. I hope to add some Watson guidance. 

[![IMAGE ALT TEXT](http://img.youtube.com/vi/NQA16nbqxls/0.jpg)](http://www.youtube.com/watch?v=NQA16nbqxls "R4 Droid Speak Speech Recognition Demo")

# Purpose
I wanted to power a home built Star Wars inspired droid with their binary droid speak seen in the movies. I wanted a real experience with natural language understanding and keyword spotting. To achieve this, I employ Windows Speech Recognition and Speech Studio custom keywords to recognize when I’m talking to the droid, e.g., “R2 what is your name?” Once the keyword is detected, a recording of the sound for an adjustable duration is collected. The sound sample is submitted to IBM Speech to Text services and the text output is submitted to IBM Watson Assistant for natural language understanding. The returned payload is parsed by the code for commands to execute locally and for sound output. I use the “pronouncing” module in python to break the returned text output into one (1) of thirty-nine (39) phonemes by breaking it into syllables and assigning each syllable a frequency. The frequency is submitted to the Windows Beep API for beeping audio output. 

You can easily replace IBM Watson with Google DialogFlow or AIML or any number of other things. You could use PyAIML to have a completely offline experience. You could connect this to an E*Z Robot Arc installation to add personality to remotely controlled robots. You could replace the Beep API with prerecorded sounds for authenticity if you prefer. The sky is the limit.

# Notes
This code adapts the Microsoft Speech Custom Keyword python sample available through the SDK.

https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/custom-keyword-basics?pivots=programming-language-python 

Additionally, this utilizes IBM Watson Assistant Lite (Free) and IBM Speech to Text Lite (Free) services through the Watson python module. 

https://pypi.org/project/ibm-watson/

https://www.ibm.com/cloud 

# Prerequisites

## Python
Known to support Python 3.6, other versions not tested but may work.

## Install Pronouncing 
https://pypi.org/project/pronouncing/ 

## Install Watson 
https://pypi.org/project/ibm-watson/

## Install Sounddevice
https://pypi.org/project/sounddevice/ 

## Install Winsound
Built in.
https://docs.python.org/3/library/winsound.html

# Edits
## droid_speech.py
You must enter your IBM Watson API (IAM) information in the following lines. 

`authenticator = IAMAuthenticator('<WATSON_API_KEY>')`

`agent_id = '<WATSON_AGENT_ID>'`

`service.set_service_url('<WATSON_SERVICE_URL>')`

`ttsauthenticator = IAMAuthenticator('SPEECH_TO_TEXT_API_KEY')`

`ttsservice.set_service_url('SPEECH_TO_TEXT_SERVICE_URL')`

`speech_key, service_region = "<SPEECH_TO_TEXT_API_KEY>", "<DATACENTER>"`

You must select which droid keyword you want. (R2, BB8, etc.) 

Edit the table name to point to the included pretrained models. By default, the droid 

### Function speech_recognize_keyword_locally_from_microphone()

```
    # Creates an instance of a keyword recognition model. Update this to
    # point to the location of your keyword recognition model.
    model = speechsdk.KeywordRecognitionModel("r4.table")

    # The phrase your keyword recognition model triggers on.
    keyword = "R4"
```
You could/should adjust the filters that remove the keyword(s) from the payload before it is sent to Watson Assistant. This helps with accuracy but isn’t required. Some of these filters will emerge though reviewing the analytics in Watson Assistant. 

` str = str.replace('are four ','').replace('Are Four ', '').replace('are Four ', '').replace('are 4 ', '').replace('our four ', '').replace('Our Four ', '').replace('our 4 ', '').replace('r 4 ', '').replace('R. for ', '')`

# Running

`python.exe main.py`
