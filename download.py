import cv2 as cv
import os
import yt_dlp
import sys

link = sys.argv[1]

# Save youtube video in mp4 format using https://github.com/yt-dlp/yt-dlp. The file name is the video's ID. To watch the video on Youtube, use the link: https://www.youtube.com/watch?v={id}
ydl_opts = { 'format' : '22+bestaudio', 'outtmpl': 'videos/%(id)s.%(ext)s', 'ignoreerrors': True} #.%(ext)s para o outtml funcionar
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([link])
