from moviepy.editor import VideoFileClip
import sys
import matplotlib.pyplot as plt
from pydub import AudioSegment

filename = sys.argv[1]
type = sys.argv[2]

if type == "1":
    video_path = f'../videos_br/{filename}.mp4'

    audio_path = f'{filename}.wav'

    video = VideoFileClip(video_path)

    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
else:
    audio = AudioSegment.from_file(f"{filename}.wav", format="wav")  

    start_time = 98000   
    end_time = 115000    

    # Cortar o áudio
    cut_audio = audio[start_time:end_time]

    # Exportar o áudio cortado
    cut_audio.export(f"{filename}_cut.wav", format="wav") 




