import whisper

# Load the Whisper model (choose: tiny, base, small, medium, large)
model = whisper.load_model("turbo")

# Path to your video file
sample = r"D:\UTS\Semester 3\Deep Learning\Assignment 3\PresentPerfect\SampleVideos\Video3.mp4"

# Transcribe the video
result = model.transcribe(sample)

# Print the full transcription
print("\nFull transcription:\n")
print(result["text"])

# Print segments with timestamps
print("\nSegments with timestamps:\n")
for segment in result["segments"]:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    print(f"[{start:.2f}s - {end:.2f}s] {text}")