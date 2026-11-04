"""Builds the internal-testing reference clip from the CC BY 4.0 VCTK corpus. The clip is never committed."""
import io, sys
import numpy as np, soundfile as sf

def build(path, speaker, min_seconds):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    shards = sorted(fs.glob("datasets/CSTR-Edinburgh/vctk@refs/convert/parquet/main/train/*.parquet"))
    clips, total, sr = [], 0.0, 48000
    for s in shards:
        with fs.open(s, "rb") as f:
            pf = pq.ParquetFile(f)
            for g in range(pf.num_row_groups):
                ids = set(pf.read_row_group(g, columns=["speaker_id"]).column(0).to_pylist())
                if speaker not in ids:
                    continue
                for r in pf.read_row_group(g, columns=["speaker_id", "audio"]).to_pylist():
                    if r["speaker_id"] != speaker:
                        continue
                    x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                    if x.ndim > 1:
                        x = x[:, 0]
                    clips += [x, np.zeros(int(sr * 0.2), dtype="float32")]
                    total += len(x) / sr + 0.2
                    if total >= min_seconds:
                        break
                if total >= min_seconds:
                    break
        if total >= min_seconds:
            break
    if total < min_seconds:
        sys.exit("could not build reference clip")
    sf.write(path, np.concatenate(clips), sr)
    return total
