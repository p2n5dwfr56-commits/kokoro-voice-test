import time, sys, torch, soundfile as sf, traceback
from round4_text import TEXT
INSTRUCT = ("An adult man, roughly forty, with a soft, understated, everyday British accent and only a subtle "
            "Midlands touch. Relaxed, calm and conversational, like a knowledgeable mate explaining football "
            "over a pint. Low-key and matter-of-fact, no excitement, no commentator or presenter delivery, "
            "no dramatic pauses, natural connected speech at an easy pace.")
try:
    from qwen_tts import Qwen3TTSModel
    t = time.time()
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cpu", dtype=torch.float32)
    print("loaded in", round(time.time()-t, 1), flush=True); t = time.time()
    wavs, sr = m.generate_voice_design(text=TEXT, language="English", instruct=INSTRUCT)
    print(f"generated {len(wavs[0])/sr:.1f}s audio in {time.time()-t:.1f}s", flush=True)
    sf.write("qwen3_voicedesign.wav", wavs[0], sr)
except Exception:
    traceback.print_exc(); sys.exit(1)
