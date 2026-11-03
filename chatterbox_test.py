# Chatterbox test: NO voice cloning, NO reference audio. Built-in default voice only.
import time, sys, traceback, torch, torchaudio as ta
TEXT = ("Right, so when a team's playing a high press, what you're really looking for is the trigger. "
        "Say the double pivot gets pinned back, and the false nine drops off into midfield. "
        "That's your cue to go. Don't just chase the ball, press with a plan.")
print("torch", torch.__version__, "threads", torch.get_num_threads(), flush=True)
which = sys.argv[1]
t = time.time()
try:
    if which == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS as M
    else:
        from chatterbox.tts import ChatterboxTTS as M
    model = M.from_pretrained(device="cpu")
    print(which, "loaded in", round(time.time()-t, 1), "s", flush=True)
    t = time.time()
    wav = model.generate(TEXT)
    dur = wav.shape[-1] / model.sr
    print(which, f"generated {dur:.1f}s audio in {time.time()-t:.1f}s", flush=True)
    ta.save(f"chatterbox_{which}.wav", wav, model.sr)
except Exception:
    traceback.print_exc(); sys.exit(1)
