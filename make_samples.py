# Kokoro voice test, round 2. Writes WAVs for each voice and each script version.
import time, numpy as np, soundfile as sf
from kokoro import KPipeline

SR = 24000
SHORT, LONG = 0.18, 0.75   # seconds. LONG only where a tactical animation needs a beat.

# (text, pause_after)  -- original approved wording
ORIGINAL = [
 ("Blue are defending in the middle third.", SHORT),
 ("They're organised, rather than chasing every Red player.", SHORT),
 ("Red are moving the ball towards their right winger, number seven.", SHORT),
 ("Watch the receiver, and the nearest Blue midfielder, number eleven.", LONG),
 ("On the first receiving touch, Red seven controls the ball and has time to pass.", SHORT),
 ("Blue eleven shouldn't automatically sprint in.", LONG),
 ("But now, compare a heavier touch.", LONG),
 ("Red seven must chase the ball before choosing the next pass.", SHORT),
 ("The opportunity has changed.", SHORT),
 ("This is a much better moment for Blue eleven to apply pressure.", 0.0),
]

# Conversational rewrite, same coaching meaning
REVISED = [
 ("Blue are defending in the middle third, and they're staying organised.", SHORT),
 ("They're not chasing every Red player.", SHORT),
 ("Red are working the ball out to their right winger, number seven.", SHORT),
 ("Keep your eye on two players: the one receiving it, and Blue's nearest midfielder, number eleven.", LONG),
 ("First touch is a good one. Red seven's got it under control, and there's time to pass.", SHORT),
 ("So Blue eleven doesn't need to sprint in.", LONG),
 ("Now, watch what happens with a heavier touch.", LONG),
 ("Red seven has to chase the ball down before he can pick his next pass.", SHORT),
 ("So the picture's changed.", SHORT),
 ("And this is a much better moment for Blue eleven to go and press.", 0.0),
]

pipe = KPipeline(lang_code='b')
SPEED = 1.0
for voice in ['bm_fable', 'bm_daniel', 'bm_lewis']:
    for name, lines in [('original', ORIGINAL), ('revised', REVISED)]:
        t = time.time(); parts = []
        for text, pause in lines:
            for _, _, audio in pipe(text, voice=voice, speed=SPEED):
                parts.append(np.asarray(audio, dtype=np.float32))
            if pause:
                parts.append(np.zeros(int(SR * pause), dtype=np.float32))
        out = np.concatenate(parts)
        sf.write(f'{voice}_{name}.wav', out, SR)
        print(voice, name, f'{len(out)/SR:.1f}s audio in {time.time()-t:.1f}s', flush=True)
