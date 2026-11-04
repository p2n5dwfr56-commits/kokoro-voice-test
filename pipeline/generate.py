#!/usr/bin/env python3
"""Football Decoded narration pipeline, proof of concept (internal testing only).

  python pipeline/generate.py --plan                 which cues would be regenerated (no model needed)
  python pipeline/generate.py                        regenerate only cues whose fingerprint changed
  python pipeline/generate.py --determinism-check    also synthesise one cue repeatedly and compare
  python pipeline/generate.py --set-review ID STATUS record a human review decision (bound to the fingerprint)
"""
import argparse, copy, hashlib, inspect, json, os, random, re, subprocess, sys, time, platform
from importlib import metadata
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
OUT, STATE, WORK = ROOT / "out", ROOT / "state", ROOT.parent / "work"
REF_PATH = ROOT.parent / "ref" / "newcastle_ref.wav"


def sha256_bytes(b): return hashlib.sha256(b).hexdigest()
def sha256_file(p): return sha256_bytes(Path(p).read_bytes())
def canon(o): return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def pkg(n):
    try: return metadata.version(n)
    except metadata.PackageNotFoundError: return None
def load(p, default=None):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else default
def save(p, o): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(o, indent=2, ensure_ascii=False) + "\n")


# ---------- text ----------
def spoken_text(text, subs):
    for s in subs["substitutions"]:
        text = re.sub(s["pattern"], s["say"], text)
    return text

DIGITS = "zero one two three four five six seven eight nine".split()
def norm(t):
    t = t.lower().replace("-", " ")
    t = re.sub(r"\d", lambda m: " " + DIGITS[int(m.group())] + " ", t)
    return re.sub(r"[^a-z ]", " ", t).split()

def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw)); prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


# ---------- fingerprint ----------
def fingerprint(spoken, ref_sha, cfg, rev, versions):
    return sha256_bytes(canon({
        "spoken_text": spoken, "reference_sha256": ref_sha,
        "model": {"hf_repo": cfg["model"]["hf_repo"], "hf_revision": rev},
        "versions": versions, "settings": cfg["settings"], "postprocess": cfg["postprocess"],
    }).encode())


# ---------- audio ----------
def trim_pad(x, sr, pp):
    idx = np.where(np.abs(x) > 10 ** (pp["trim_threshold_db"] / 20))[0]
    core = x[idx[0]: idx[-1] + 1] if len(idx) else x
    pad = np.zeros(int(sr * pp["pad_ms"] / 1000), dtype="float32")
    return np.concatenate([pad, core, pad])

def ff_measure(path, extra=""):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                        f"loudnorm=I=-16:TP=-1.5:LRA=11{extra}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stderr[r.stderr.rindex("{"): r.stderr.rindex("}") + 1])

def encode(x, sr, pp, mp3_path):
    WORK.mkdir(exist_ok=True); tmp = WORK / "tmp.wav"
    sf.write(tmp, trim_pad(x, sr, pp), sr, subtype="FLOAT")
    m = ff_measure(tmp)
    f = (f"loudnorm=I={pp['target_lufs']}:TP={pp['true_peak_target_dbtp']}:LRA={pp['lra']}"
         f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
         f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(tmp), "-af", f,
                    "-ar", str(pp["sample_rate"]), "-ac", str(pp["channels"]), "-c:a", "libmp3lame",
                    "-b:a", pp["mp3_bitrate"], "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
                    str(mp3_path)], check=True)
    fin = ff_measure(mp3_path)  # measure the delivered MP3 itself
    lufs, tp = float(fin["input_i"]), float(fin["input_tp"])
    return {"integrated_lufs": round(lufs, 2), "true_peak_dbtp": round(tp, 2),
            "loudness_ok": abs(lufs - pp["target_lufs"]) <= pp["lufs_tolerance"] and tp <= pp["true_peak_ceiling_dbtp"]}

def duration(path): return round(sf.info(str(path)).duration, 3)


# ---------- synthesis ----------
class Synth:
    def __init__(self, cfg, rev, fake=False):
        self.cfg, self.fake, self.rev = cfg, fake, rev
        self.s = cfg["settings"]
        if fake:
            self.sr = 24000; return
        import torch
        torch.set_num_threads(self.s["torch_threads"])
        from huggingface_hub import snapshot_download
        from chatterbox.tts import ChatterboxTTS
        path = snapshot_download(repo_id=cfg["model"]["hf_repo"], revision=rev)
        self.model = ChatterboxTTS.from_local(path, device="cpu")
        self.sr = self.model.sr
        accepted = inspect.signature(self.model.generate).parameters
        self.kw = {k: self.s[k] for k in ("exaggeration", "cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p")
                   if k in accepted and k in self.s}

    def __call__(self, text, seed):
        if self.fake:
            rng = np.random.default_rng(seed); n = int(self.sr * 0.33 * len(text.split()))
            t = np.arange(n) / self.sr
            return (rng.standard_normal(n) * 0.2 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t))).astype("float32")
        import torch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        w = self.model.generate(text, audio_prompt_path=str(REF_PATH), **self.kw)
        return w.squeeze().cpu().numpy().astype("float32")


class ASR:
    def __init__(self, qa, enabled=True):
        self.on = enabled
        if enabled:
            from faster_whisper import WhisperModel
            self.m = WhisperModel(qa["asr_model"], device="cpu", compute_type="int8")
    def __call__(self, path):
        if not self.on: return None
        segs, _ = self.m.transcribe(str(path), language="en", beam_size=5)
        return " ".join(s.text.strip() for s in segs)


def qa_check(expected, transcript, dur, qa, loud):
    words = len(norm(expected)); wps = words / dur if dur else 0
    w = None if transcript is None else wer(expected, transcript)
    terms = [t for t in qa["key_terms"] if t in " ".join(norm(expected))]
    terms_heard = None if transcript is None else {t: (t in " ".join(norm(transcript))) for t in terms}
    ok = (w is None or w <= qa["max_wer"]) and qa["words_per_second_min"] <= wps <= qa["words_per_second_max"] \
         and loud["loudness_ok"] and (terms_heard is None or all(terms_heard.values()))
    return {"transcript": transcript, "wer": None if w is None else round(w, 3), "words_per_second": round(wps, 2),
            "key_terms_heard": terms_heard, "passed": bool(ok)}


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "cues.generic.json"))
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--only"); ap.add_argument("--determinism-check", action="store_true")
    ap.add_argument("--override", action="append", default=[], help="settings.key=value, for --plan what-if only")
    ap.add_argument("--fake-synth", action="store_true", help="developer test: no model, noise instead of speech")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--set-review", nargs=2, metavar=("ID", "STATUS"))
    a = ap.parse_args()

    cfg = load(ROOT / "config.json"); subs = load(ROOT / "pronunciations.json")
    manifest = load(a.manifest); idx = load(OUT / "index.json", {"cues": []})
    review = load(STATE / "review.json", {})
    old = {c["id"]: c for c in idx["cues"]}

    if a.set_review:
        cid, status = a.set_review
        review[cid] = {"status": status, "fingerprint": old[cid]["fingerprint"], "date": time.strftime("%Y-%m-%d")}
        save(STATE / "review.json", review); old[cid]["review_status"] = status
        save(OUT / "index.json", idx); print("review recorded", review[cid]); return

    for o in a.override:  # what-if only
        k, v = o.split("=", 1); cfg["settings"][k.split(".")[-1]] = type(cfg["settings"][k.split(".")[-1]])(v)
        if not a.plan: sys.exit("--override is for --plan only")

    t_start = time.time(); report = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "timings_s": {}, "host": {
        "python": platform.python_version(), "cpu_count": os.cpu_count(), "platform": platform.platform()}}

    # model revision: pinned in config, else in state file, else resolved once and recorded
    rev = cfg["model"]["hf_revision"]
    rev_file = STATE / "model_revision.txt"
    if not rev and rev_file.exists(): rev = rev_file.read_text().strip()
    if not rev:
        if a.fake_synth: rev = "fake-revision"
        else:
            from huggingface_hub import HfApi
            rev = HfApi().model_info(cfg["model"]["hf_repo"]).sha
        if not a.plan and not a.fake_synth: STATE.mkdir(exist_ok=True); rev_file.write_text(rev + "\n")
    versions = {"chatterbox-tts": pkg("chatterbox-tts"), "torch": pkg("torch"), "numpy": pkg("numpy"),
                "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split()[2]}

    # reference
    ref_file = STATE / "reference.sha256"
    t = time.time()
    if a.plan or a.fake_synth:
        ref_sha = ref_file.read_text().strip() if ref_file.exists() else "unknown-reference"
    else:
        if not REF_PATH.exists():
            REF_PATH.parent.mkdir(exist_ok=True)
            import build_reference
            print("building reference clip ...", flush=True)
            build_reference.build(str(REF_PATH), cfg["reference"]["speaker"], cfg["reference"]["min_seconds"])
        ref_sha = sha256_file(REF_PATH)
        report["timings_s"]["reference_build_or_cache"] = round(time.time() - t, 1)
        STATE.mkdir(exist_ok=True); ref_file.write_text(ref_sha + "\n")

    cues = [c for c in manifest["cues"] if not a.only or c["id"] == a.only]
    plan = []
    for c in manifest["cues"]:
        sp = spoken_text(c["text"], subs); fp = fingerprint(sp, ref_sha, cfg, rev, versions)
        o = old.get(c["id"]); mp3 = OUT / f"{c['id']}.mp3"
        reason = "new cue" if not o else "fingerprint changed" if o["fingerprint"] != fp else \
                 "mp3 missing or altered" if not mp3.exists() or sha256_file(mp3) != o["sha256"] else None
        plan.append((c, sp, fp, reason))
    todo = [p for p in plan if p[3] and (not a.only or p[0]["id"] == a.only)]
    print(f"PLAN: {len(todo)} of {len(plan)} cues to (re)generate; model {rev[:12]}, ref {ref_sha[:12]}")
    for c, _, fp, r in plan: print(f"  {c['id']:<18} {fp[:12]}  {'REGENERATE (' + r + ')' if r else 'unchanged'}")
    if a.plan: return

    OUT.mkdir(exist_ok=True)
    new_entries, attempts_log = {}, {}
    if todo or a.determinism_check:
        t = time.time(); synth = Synth(cfg, rev, a.fake_synth); asr = ASR(cfg["qa"], not (a.no_asr or a.fake_synth))
        report["timings_s"]["model_load"] = round(time.time() - t, 1)
    pp, qa, base = cfg["postprocess"], cfg["qa"], cfg["settings"]["base_seed"]

    for c, sp, fp, reason in todo:
        best = None; t_cue = time.time(); gen_t = 0
        for attempt in range(qa["max_attempts"]):
            seed = base + attempt; t = time.time()
            x = synth(sp, seed); gen_t += time.time() - t
            mp3 = OUT / f"{c['id']}.mp3"; loud = encode(x, synth.sr, pp, mp3)
            d = duration(mp3); tr = asr(mp3); q = qa_check(sp, tr, d, qa, loud)
            print(f"  {c['id']} attempt {attempt+1} seed {seed}: {d}s, WER {q['wer']}, {loud['integrated_lufs']} LUFS, "
                  f"TP {loud['true_peak_dbtp']}, passed={q['passed']}", flush=True)
            cand = {"seed_used": seed, "attempt": attempt + 1, "duration_s": d, **loud, "qa": q, "mp3_bytes": mp3.read_bytes()}
            if best is None or (q["passed"] and not best["qa"]["passed"]) or \
               (q["passed"] == best["qa"]["passed"] and (q["wer"] or 0) < (best["qa"]["wer"] or 0)): best = cand
            if q["passed"]: break
        (OUT / f"{c['id']}.mp3").write_bytes(best.pop("mp3_bytes"))
        rv = review.get(c["id"]); status = rv["status"] if rv and rv["fingerprint"] == fp else "unreviewed"
        new_entries[c["id"]] = {
            "id": c["id"], "kind": c["kind"], "sync_event": c["sync_event"], "file": f"out/{c['id']}.mp3",
            "text_source": c["text"], "text_spoken": sp, "fingerprint": fp,
            "sha256": sha256_file(OUT / f"{c['id']}.mp3"), "duration_s": best["duration_s"],
            "integrated_lufs": best["integrated_lufs"], "true_peak_dbtp": best["true_peak_dbtp"], "loudness_ok": best["loudness_ok"],
            "qa": best["qa"], "qa_status": "passed" if best["qa"]["passed"] else "NEEDS_ATTENTION",
            "review_status": status,
            "generation": {"model_hf_repo": cfg["model"]["hf_repo"], "model_revision": rev, "versions": versions,
                           "settings": {**{k: v for k, v in cfg["settings"].items() if k != "base_seed"},
                                        "base_seed": base, "seed_used": best["seed_used"], "attempts": best["attempt"]},
                           "reference_sha256": ref_sha, "pronunciation_version": subs["version"],
                           "generation_seconds": round(gen_t, 1), "total_seconds": round(time.time() - t_cue, 1),
                           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
        report["timings_s"][c["id"]] = {"audio_s": best["duration_s"], "synth_s": round(gen_t, 1),
                                        "realtime_factor": round(gen_t / best["duration_s"], 1), "attempts": best["attempt"]}

    # keep unchanged entries, refresh review status, drop cues removed from the manifest
    final = []
    for c, sp, fp, reason in plan:
        e = new_entries.get(c["id"]) or old.get(c["id"])
        if e and not new_entries.get(c["id"]):
            rv = review.get(c["id"]); e["review_status"] = rv["status"] if rv and rv["fingerprint"] == fp else "unreviewed"
        if e: final.append(e)
    live = {e["id"] for e in final}
    for f in OUT.glob("*.mp3"):
        if f.stem not in live: f.unlink(); print("removed orphan", f.name)

    if a.determinism_check:
        c, sp, fp, _ = min(plan, key=lambda p: len(p[1])); res = {}
        xs = [synth(sp, base), synth(sp, base), synth(sp, base + 1)]
        res["cue"] = c["id"]
        res["same_seed_waveform_identical"] = bool(np.array_equal(xs[0], xs[1]))
        res["same_seed_max_abs_diff"] = float(np.max(np.abs(xs[0] - xs[1][: len(xs[0])])) if len(xs[0]) == len(xs[1]) else -1)
        res["same_seed_same_length"] = len(xs[0]) == len(xs[1])
        shas = []
        for i, x in enumerate(xs[:2]):
            p = WORK / f"det{i}.mp3"; encode(x, synth.sr, pp, p); shas.append(sha256_file(p))
        res["same_seed_mp3_identical"] = shas[0] == shas[1]
        res["different_seed_waveform_identical"] = bool(len(xs[0]) == len(xs[2]) and np.array_equal(xs[0], xs[2]))
        report["determinism_check"] = res; print("DETERMINISM", json.dumps(res))

    idx = {"pipeline_version": cfg["pipeline_version"], "manifest": Path(a.manifest).name,
           "internal_testing_only": True, "cues": final}
    save(OUT / "index.json", idx)
    report.update({"regenerated": [c["id"] for c, *_ in todo], "skipped": [c["id"] for c, _, _, r in plan if not r],
                   "failures": [e["id"] for e in final if e["qa_status"] != "passed"]})
    report["timings_s"]["total"] = round(time.time() - t_start, 1)
    save(OUT / "run-report.json", report)
    print(json.dumps({k: report[k] for k in ("regenerated", "skipped", "failures")}), "total", report["timings_s"]["total"], "s")

if __name__ == "__main__":
    main()
