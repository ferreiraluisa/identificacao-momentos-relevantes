import whisper
import time
import sys
from pydub import AudioSegment
from pydub.playback import play

filename = sys.argv[1]
model = whisper.load_model("base")
result = model.transcribe("audios/" + filename, language="portuguese", temperature=0.2)


start_time = 0
for segment in result["segments"]:
    start_time = abs(segment["start"] - start_time)
    time.sleep(start_time)
    print(segment["text"])
    time.sleep(segment["end"] - segment["start"])
    start_time = segment["end"]


# define a palavra que você quer procurar
palavras = ["peça", "polícia", "arma", "assalto", "machucado"]
# print(result['text'])
# itera sobre a lista de tokens e verifica se a palavra está presente
for token in result["segments"]:
    for palavra in palavras:
        if palavra in token['text']:
            print(f"{token['text']} foi falada no tempo {token['start']:.2f} - {token['end']:.2f} segundos")

