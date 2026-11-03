# Chatterbox (original 500M) with a CC BY-SA 4.0 licensed reference clip (Google UK/Ireland English, OpenSLR 83, Midlands male).
import glob, time, sys, traceback, soundfile as sf, torchaudio as ta
from round4_text import TEXT
clips = sorted(glob.glob("ref/**/*.wav", recursive=True))
print("reference clips found:", len(clips))
ref = None
for c in clips:
    i = sf.info(c); d = i.frames / i.samplerate
    if 7 <= d <= 12: ref = c; print("using clip", c.split('/')[-1], f"{d:.1f}s"); break
if not ref: sys.exit("no suitable clip")
try:
    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained(device="cpu"); t = time.time()
    wav = m.generate(TEXT, audio_prompt_path=ref, exaggeration=0.3, cfg_weight=0.3)
    print(f"generated {wav.shape[-1]/m.sr:.1f}s audio in {time.time()-t:.1f}s", flush=True)
    ta.save("chatterbox_midlands_reference.wav", wav, m.sr)
except Exception:
    traceback.print_exc(); sys.exit(1)
