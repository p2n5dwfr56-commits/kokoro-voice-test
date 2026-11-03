# Piper (free, offline) British voices, revised When to Press script.
import json, subprocess, sys, urllib.request, pathlib
BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/"
TEXT = """Blue are defending in the middle third, and they're staying organised.
They're not chasing every Red player.
Red are working the ball out to their right winger, number seven.
Keep your eye on two players: the one receiving it, and Blue's nearest midfielder, number eleven.
First touch is a good one. Red seven's got it under control, and there's time to pass.
So Blue eleven doesn't need to sprint in.
Now, watch what happens with a heavier touch.
Red seven has to chase the ball down before he can pick his next pass.
So the picture's changed.
And this is a much better moment for Blue eleven to go and press.
"""
# VCTK speaker ids (accent labels are from memory of the VCTK corpus notes, UNVERIFIED -- judge by ear)
VCTK = {"p243": "London", "p274": "Essex", "p256": "Birmingham", "p259": "Nottingham",
        "p232": "Southern England", "p254": "Surrey", "p258": "Southern England", "p279": "Oxford"}
def fetch(sub, name):
    for ext in (".onnx", ".onnx.json"):
        urllib.request.urlretrieve(f"{BASE}{sub}/{name}{ext}", f"{name}{ext}")
def say(model, out, speaker=None):
    cmd = [sys.executable, "-m", "piper", "-m", f"{model}.onnx", "-f", out, "--sentence-silence", "0.2"]
    if speaker is not None: cmd += ["-s", str(speaker)]
    subprocess.run(cmd, input=TEXT.encode(), check=True)
for sub, name in [("northern_english_male/medium", "en_GB-northern_english_male-medium"), ("alan/medium", "en_GB-alan-medium")]:
    fetch(sub, name); say(name, f"piper_{name.split('-')[1]}.wav")
fetch("vctk/medium", "en_GB-vctk-medium")
ids = json.load(open("en_GB-vctk-medium.onnx.json"))["speaker_id_map"]
print("vctk speakers available:", len(ids))
for sp, label in VCTK.items():
    if sp in ids: say("en_GB-vctk-medium", f"piper_vctk_{sp}_{label.replace(' ','')}.wav", ids[sp])
    else: print("missing", sp)
