import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()

def speak(text):
    print(f"Robot says: {text}")
    engine.say(text)
    engine.runAndWait()

def listen_and_respond():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[Listening] Speak a command ('move forward', 'stop', 'turn left')...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"User Command: '{command}'")
        
        if "forward" in command:
            speak("Moving forward at half speed.")
        elif "stop" in command:
            speak("Emergency stop executed!")
        elif "left" in command:
            speak("Turning left 90 degrees.")
        else:
            speak(f"Command '{command}' not recognized.")
    except sr.UnknownValueError:
        speak("Sorry, I could not understand the speech.")
    except sr.RequestError:
        speak("Speech recognition service is offline.")

if __name__ == "__main__":
    listen_and_respond()
