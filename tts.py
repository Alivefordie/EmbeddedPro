from gtts import gTTS

tts = gTTS("เรามาช่วยแล้ว", lang="th")
tts.save("help.mp3")
