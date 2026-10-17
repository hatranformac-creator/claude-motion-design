"""AIMarketing 15s soundtrack, 100 % synthesized (Mixkit is unreachable from this container).
120 BPM, beat = 0.5 s. Soft Apple-bumper bed: pad from the start, pulse from the Create card (2.4 s),
everything drops at 11.95 s into a riser, logo hit on beat 25 (12.5 s). SFX placed on the exact animation times.
-> out/mix.wav (two-pass loudnorm -14 LUFS, TP -1.5)
"""
import json, subprocess
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.io import wavfile

HERE = Path(__file__).parent
SR, T = 48000, 15.0
N = int(SR * T)
mix = np.zeros((N, 2))
rnd = np.random.RandomState(11)
BEAT = 0.5


def tt(n): return np.arange(n) / SR
def lp(x, f, o=2): return sosfilt(butter(o, f, 'low', fs=SR, output='sos'), x)
def hp(x, f, o=2): return sosfilt(butter(o, f, 'high', fs=SR, output='sos'), x)
def bp(x, a, b, o=2): return sosfilt(butter(o, [a, b], 'band', fs=SR, output='sos'), x)
def midi(m): return 440 * 2 ** ((m - 69) / 12)


def add(sig, t0, gain=1.0, pan=0.0):
    i = int(round(t0 * SR))
    if i >= N: return
    if i < 0: sig, i = sig[-i:], 0
    sig = sig[:N - i]
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    mix[i:i + len(sig), 0] += sig * gain * l * 1.414
    mix[i:i + len(sig), 1] += sig * gain * r * 1.414


def env(n, a, d):  # attack, exponential decay (s)
    x = tt(n); return np.minimum(1, x / max(a, 1e-4)) * np.exp(-x / d)


# ---------------- music ----------------
CHORDS = [[48, 55, 59, 62, 64], [45, 52, 55, 59, 60], [41, 48, 52, 55, 57], [43, 50, 55, 57, 62]]  # Cmaj9 Am9 Fmaj9 G6/9


def pad(notes, dur, gain):
    n = int(dur * SR); x = tt(n); s = np.zeros(n)
    for m in notes:
        for det in (-0.07, 0.07):
            f = midi(m + 12) * (1 + det / 100 * 12)
            s += np.sin(2 * np.pi * f * x) + .35 * np.sin(2 * np.pi * 2 * f * x) + .12 * np.sin(2 * np.pi * 3 * f * x)
    s = lp(s, 1600)
    a = np.minimum(1, x / .5) * np.minimum(1, (dur - x) / .5)
    return s * a * gain / len(notes)


for bar in range(6):  # 0..12 s
    t0 = bar * 2 - .25
    add(pad(CHORDS[bar % 4], 2.5, .09 if bar else .07), max(0, t0), pan=0)

# pulse from the Create card to the end of Measure
def kick():
    n = int(.32 * SR); x = tt(n)
    f = 48 + 70 * np.exp(-x / .03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x / .11)


def hat():
    n = int(.05 * SR); return hp(rnd.randn(n), 7000) * np.exp(-tt(n) / .012)


def pluck(m, d=.28):
    n = int(d * 1.6 * SR); x = tt(n); f = midi(m)
    s = np.sin(2 * np.pi * f * x) + .3 * np.sin(2 * np.pi * 2 * f * x) * np.exp(-x / .05)
    return s * env(n, .003, d)


for b in range(5, 24):  # beats 5..23 (2.5 s .. 11.5 s)
    add(kick(), b * BEAT, .17)
    if b >= 10: add(hat(), b * BEAT + .25, .05, pan=.25)
ARP = [0, 2, 4, 3, 1, 3, 4, 2]
for k in range(2 * 5, 2 * 24):  # 8th-note arp of chord tones
    t0 = k * BEAT / 2
    ch = CHORDS[int(t0 // 2) % 4]
    add(pluck(ch[ARP[k % 8]] + 24), t0, .045 if t0 < 5 else .06, pan=-.3 if k % 2 else .3)

# riser into the logo (11.5 -> 12.48), then the hit at 12.5
n = int(.98 * SR); x = tt(n)
rs = bp(rnd.randn(n), 800, 6000) * (x / x[-1]) ** 2.2
sw = np.sin(2 * np.pi * np.cumsum(300 + 900 * (x / x[-1]) ** 2) / SR) * (x / x[-1]) ** 3
add(rs * .14 + sw * .04, 11.5)


def bell(m, d=2.6):
    n = int(d * SR); x = tt(n); f = midi(m); s = np.zeros(n)
    for ratio, g, dec in [(1, 1, 1.4), (2, .35, .8), (2.76, .18, .5), (4.07, .08, .3)]:
        s += g * np.sin(2 * np.pi * f * ratio * x) * np.exp(-x / dec)
    return s * np.minimum(1, x / .004)


for i, m in enumerate([60, 64, 67, 71, 74]):
    add(bell(m), 12.5 + i * .012, .055, pan=(i - 2) * .2)
n = int(1.2 * SR); x = tt(n)
add(np.sin(2 * np.pi * np.cumsum(38 + 40 * np.exp(-x / .05)) / SR) * np.exp(-x / .45), 12.5, .5)  # sub
add(pad([48, 55, 60, 64, 67, 71], 2.7, .11), 12.45)  # warm tail under the end card

# ---------------- sfx (exact animation times) ----------------
def blip(f0, f1, d=.05):
    n = int(d * SR); x = tt(n); f = np.linspace(f0, f1, n)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, .002, d / 3)


def tick(f=2200, d=.03):
    n = int(d * SR); return bp(rnd.randn(n), f * .7, f * 1.4) * env(n, .0005, .006)


def whoosh(d, f0=400, f1=3000, peak=.7):
    n = int(d * SR); x = tt(n) / d
    s = rnd.randn(n); out = np.zeros(n); step = 2048
    for i in range(0, n, step):  # moving band
        c = f0 + (f1 - f0) * (i / n)
        out[i:i + step] = bp(s[max(0, i - 4096):i + step], c * .7, min(c * 1.4, 20000))[-len(s[i:i + step]):]
    shape = np.where(x < peak, (x / peak) ** 2, np.exp(-(x - peak) / (1 - peak) * 3))
    return out * shape


def click():
    a = np.zeros(int(.055 * SR)); a[:int(.02 * SR)] += tick(3000, .02); i = int(.035 * SR); a[i:i + int(.02 * SR)] += tick(1800, .02) * .6
    return a


for i in range(9): add(blip(1500 + i * 70, 1900 + i * 70, .045), .3 + i * .085, .06, pan=-.6 + i * .15)  # chips
add(whoosh(.5, 3000, 300, .9), 1.52, .16)                    # chips pulled into the centre
add(blip(320, 760, .06), 1.86, .16)                          # dot pops
add(whoosh(.35, 500, 2500, .6), 2.18, .08)                   # dot opens into the card
for k in range(37):                                          # keystrokes (2.62 -> 3.2)
    add(tick(2600 + 600 * rnd.rand(), .025), 2.62 + k * (0.58 / 37), .05 + .03 * rnd.rand(), pan=.2)
for i in range(4): add(blip(1300 + i * 140, 1500 + i * 140, .04), 3.22 + i * .07, .05, pan=.3)  # channels
for c in (3.5, 7.0, 8.0): add(click(), c - .01, .35, pan=.35)                                   # clicks
add(bell(84, .7), 3.84, .03, pan=.2)                         # draft starts writing
for s in (5.0, 7.5, 10.0): add(whoosh(.32, 600, 2400, .55), s - .12, .07)  # section changes
for i in range(6): add(blip(1100 + i * 110, 1250 + i * 110, .05), 5.36 + i * .14, .055, pan=.3)  # QC checks
add(bell(79, .9), 6.5, .05); add(bell(84, .9), 6.56, .045)   # passed QC
for k in range(5): add(blip(900 + k * 120, 1300 + k * 120, .05), 8.2 + k * .075, .06, pan=-.4 + k * .2)  # channel cards
for i in range(6): add(tick(4200, .02), 8.62 + i * .07, .08, pan=-.3 + i * .12)                         # live badges
add(whoosh(.32, 2600, 500, .5), 9.98, .06)                   # cards fold back
add(whoosh(.45, 2400, 300, .7), END := 11.95, .1)            # card shrinks into the dot

# ---------------- master ----------------
fade = np.ones(N); fl = int(.8 * SR); fade[-fl:] = np.linspace(1, 0, fl) ** 1.5
mix *= fade[:, None]
mix /= np.abs(mix).max() * 1.05
(HERE / "out").mkdir(exist_ok=True)
raw = HERE / "out/raw.wav"
wavfile.write(raw, SR, (mix * 32767).astype(np.int16))
r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(raw), "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                   capture_output=True, text=True)
m = json.loads(r.stderr[r.stderr.rfind("{"):r.stderr.rfind("}") + 1])
af = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
      f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,aresample=48000")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af", af, "-ar", "48000", str(HERE / "out/mix.wav")], check=True)
r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(HERE / "out/mix.wav"), "-af", "loudnorm=print_format=json", "-f", "null", "-"],
                   capture_output=True, text=True)
m2 = json.loads(r.stderr[r.stderr.rfind("{"):r.stderr.rfind("}") + 1])
print("mix ->", HERE / "out/mix.wav", "| LUFS", m2["input_i"], "TP", m2["input_tp"])
