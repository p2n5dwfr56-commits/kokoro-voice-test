# Kokoro-82M voice test. Run: python3.11 -m pip install kokoro soundfile && python3.11 make_samples.py
# (Linux/macOS also need espeak-ng: apt install espeak-ng / brew install espeak-ng)
import time, numpy as np, soundfile as sf
from kokoro import KPipeline
SCRIPT = ("Blue are defending in the middle third. They're organised, rather than chasing every Red player.\n"
"Red are moving the ball towards their right winger, number seven.\n"
"Watch the receiver, and the nearest Blue midfielder, number eleven.\n"
"On the first receiving touch, Red seven controls the ball and has time to pass. Blue eleven shouldn't automatically sprint in.\n"
"But now, compare a heavier touch.\n"
"Red seven must chase the ball before choosing the next pass.\n"
"The opportunity has changed.\n"
"This is a much better moment for Blue eleven to apply pressure.")
pipe = KPipeline(lang_code='b')  # British English; downloads weights from Hugging Face on first run
for voice in ['bm_george', 'bm_lewis', 'bf_emma']:
    t = time.time(); parts = []
    for _, _, audio in pipe(SCRIPT, voice=voice, speed=0.9, split_pattern=r'\n+'):
        parts.append(audio); parts.append(np.zeros(int(24000*0.45), dtype=np.float32))  # pause between sentences
    out = np.concatenate(parts); sf.write(f'{voice}.wav', out, 24000)
    print(voice, f'{len(out)/24000:.1f}s audio in {time.time()-t:.1f}s')
