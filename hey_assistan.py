import speech_recognition as sr

def sesli_komut_al():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎧 Dinleniyor... (Wake word: 'asistan')")
        audio = recognizer.listen(source, phrase_time_limit=5)

    try:
        metin = recognizer.recognize_google(audio, language='tr-TR')
        print("🗣️ Algılanan:", metin.lower())
        return metin.lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        print("Google API'ye ulaşılamadı.")
        return ""


