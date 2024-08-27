from faster_whisper import WhisperModel
import sys


model_size = "large-v3"
audio = sys.argv[1]

model = WhisperModel(model_size, device="cuda", compute_type="float16")

segments, info = model.transcribe(
    audio,
    beam_size=5,
    vad_filter=True,
)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
words = [ "run", "gun", "shot",
"ground", "hands up", "suspect", "reload", "car", "police",
"drop", "cuffs", "robbery", "victim"]

hist = []
for segment in segments:
    keywords = 0
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    for word in words:
        if word in segment.text:
            keywords += 1
    hist.append(keywords/len(segment.text.split()))
print(hist)
    