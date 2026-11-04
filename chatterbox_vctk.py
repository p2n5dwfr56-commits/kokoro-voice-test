# Chatterbox (original 500M), VCTK speaker p286 (male, Newcastle; CC BY 4.0, Univ. of Edinburgh) as reference.
# Round 6: two livelier variants. Same reference clip construction as the first Newcastle sample.
import io, os, sys, time, traceback, numpy as np, soundfile as sf, torchaudio as ta
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from round4_text import TEXT
SPK = "p286"
fs = HfFileSystem(); REV = "refs/convert/parquet"
shards = sorted(fs.glob(f"datasets/CSTR-Edinburgh/vctk@{REV}/main/train/*.parquet"))
clips, total, sr = [], 0.0, 48000
for s in shards:
    with fs.open(s, "rb") as f:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            ids = set(pf.read_row_group(g, columns=["speaker_id"]).column(0).to_pylist())
            if SPK not in ids: continue
            for r in pf.read_row_group(g, columns=["speaker_id", "audio"]).to_pylist():
                if r["speaker_id"] != SPK: continue
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if x.ndim > 1: x = x[:, 0]
                clips += [x, np.zeros(int(sr * 0.2), dtype="float32")]; total += len(x) / sr + 0.2
                if total >= 10: break
            if total >= 10: break
    if total >= 10: break
os.makedirs("ref", exist_ok=True); sf.write("ref/newcastle_ref.wav", np.concatenate(clips), sr)
print(f"reference built: {total:.1f}s", flush=True)
try:
    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained(device="cpu")
    for name, ex, cfg in [("lively1", 0.5, 0.4), ("lively2", 0.6, 0.5)]:
        t = time.time()
        wav = m.generate(TEXT, audio_prompt_path="ref/newcastle_ref.wav", exaggeration=ex, cfg_weight=cfg)
        print(f"{name} (exaggeration {ex}, cfg {cfg}): {wav.shape[-1]/m.sr:.1f}s audio in {time.time()-t:.1f}s", flush=True)
        ta.save(f"chatterbox_newcastle_{name}.wav", wav, m.sr)
except Exception:
    traceback.print_exc(); sys.exit(1)
