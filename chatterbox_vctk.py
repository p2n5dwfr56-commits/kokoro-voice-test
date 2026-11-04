# Chatterbox (original 500M) with a CC BY 4.0 reference built from VCTK (Univ. of Edinburgh): a male speaker labelled Newcastle.
import io, os, sys, time, traceback, numpy as np, soundfile as sf, torchaudio as ta
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from round4_text import TEXT
fs = HfFileSystem(); REV = "refs/convert/parquet"
shards = sorted(p for p in fs.glob(f"datasets/CSTR-Edinburgh/vctk@{REV}/main/train/*.parquet"))
print("shards:", len(shards))
meta = {}
for s in shards:
    with fs.open(s, "rb") as f:
        pf = pq.ParquetFile(f)
        if s == shards[0]: print("columns:", pf.schema_arrow.names)
        t = pf.read(columns=["speaker_id", "gender", "accent", "region"]).to_pandas().drop_duplicates()
        for r in t.itertuples(): meta[r.speaker_id] = (r.gender, r.accent, r.region)
ne = {k: v for k, v in meta.items() if any(w in str(v[2]).lower() for w in ("newcastle", "tyne", "northumb", "durham", "sunderland", "geordie"))}
print("North East labelled speakers:", ne)
male = [k for k, v in ne.items() if str(v[0]).upper().startswith("M")]
if not male: sys.exit("No male North East speaker found in VCTK -- stopping rather than substituting another voice.")
spk = male[0]; print("using speaker", spk, ne[spk])
clips, total = [], 0.0
for s in shards:
    with fs.open(s, "rb") as f:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(columns=["speaker_id", "audio"], batch_size=64):
            for r in b.to_pylist():
                if r["speaker_id"] != spk: continue
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if x.ndim > 1: x = x[:, 0]
                clips += [x, np.zeros(int(sr * 0.2), dtype="float32")]; total += len(x) / sr + 0.2
                if total >= 10: break
            if total >= 10: break
    if total >= 10: break
os.makedirs("ref", exist_ok=True); sf.write("ref/newcastle_ref.wav", np.concatenate(clips), sr)
print(f"reference built: {total:.1f}s")
try:
    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained(device="cpu"); t = time.time()
    wav = m.generate(TEXT, audio_prompt_path="ref/newcastle_ref.wav", exaggeration=0.3, cfg_weight=0.3)
    print(f"generated {wav.shape[-1]/m.sr:.1f}s audio in {time.time()-t:.1f}s", flush=True)
    ta.save("chatterbox_newcastle_reference.wav", wav, m.sr)
except Exception:
    traceback.print_exc(); sys.exit(1)
