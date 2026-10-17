"""AIMarketing 15s renderer (adapted from the skill's render_template.py).
probe t1 t2 ...   -> probe/sheet.png
chunk k n         -> render chunk k of n (subframes blended) -> out/chunk_k.mp4
concat n          -> out/video.mp4
pops              -> single-frame pop / flash scan on out/video.mp4
Serve first: python3 -m http.server 8771 --directory <this folder>
"""
import asyncio, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from playwright.async_api import async_playwright

HERE = Path(__file__).parent
FF = "ffmpeg"
URL = "http://localhost:8771/film.html"
FPS, T, SUB, SHUTTER = 60, 15.0, 5, 0.5
CUTS = []  # no hard cuts in this film
ARGS = ["--deterministic-mode", "--run-all-compositor-stages-before-draw", "--disable-threaded-animation",
        "--disable-checker-imaging", "--font-render-hinting=none", "--force-color-profile=srgb"]


async def open_page(p):
    b = await p.chromium.launch(args=ARGS, executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    pg = await b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append("console: " + m.text) if m.type in ("error", "warning") else None)
    await pg.goto(URL)
    await pg.wait_for_function("window.ready === true", timeout=120000)
    await pg.evaluate("document.fonts.ready")
    await pg.add_style_tag(content="*,*::before,*::after{transition:none!important}#hud{display:none!important}")
    await pg.evaluate("document.getAnimations().forEach(a => a.pause())")
    return b, pg, errs


async def shot(pg, t, path, fmt="png"):
    await pg.evaluate(f"window.seek({t})")
    await pg.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
    el = await pg.query_selector("#stage")
    await el.screenshot(path=str(path), type=fmt, **({"quality": 95} if fmt == "jpeg" else {}))


async def probe(times):
    out = HERE / "probe"; out.mkdir(exist_ok=True)
    for f in out.glob("p_*.png"): f.unlink()
    async with async_playwright() as p:
        b, pg, errs = await open_page(p)
        for i, t in enumerate(times): await shot(pg, t, out / f"p_{i:02d}.png")
        await b.close()
    if errs: print("PAGE ERRORS:", errs[:8])
    cols = 4 if len(times) > 4 else len(times)
    rows = -(-len(times) // cols)
    subprocess.run([FF, "-v", "error", "-y", "-i", str(out / "p_%02d.png"), "-vf",
                    f"scale=480:-1,tile={cols}x{rows}:padding=6:color=white", "-frames:v", "1", str(out / "sheet.png")], check=True)
    print("probe ->", out / "sheet.png")


async def chunk(k, n):
    total = int(round(T * FPS))
    a, b_ = k * total // n, (k + 1) * total // n
    sub = HERE / f"sub{k}"; shutil.rmtree(sub, ignore_errors=True); sub.mkdir()
    offs = [(j - (SUB - 1) / 2) * SHUTTER / (FPS * SUB) for j in range(SUB)]
    idx = 0
    async with async_playwright() as p:
        b, pg, errs = await open_page(p)
        for i in range(a, b_):
            for o in offs:
                t = min(T - 1e-3, max(0.0, i / FPS + o))
                await shot(pg, t, sub / f"s_{idx:05d}.jpg", "jpeg"); idx += 1
            if (i - a) % 60 == 0: print(f"chunk {k}: frame {i - a}/{b_ - a}", flush=True)
        await b.close()
    if errs: print("PAGE ERRORS:", errs[:8])
    (HERE / "out").mkdir(exist_ok=True)
    subprocess.run([FF, "-v", "error", "-y", "-framerate", str(FPS * SUB), "-i", str(sub / "s_%05d.jpg"),
                    "-vf", f"tmix=frames={SUB},select='eq(mod(n\\,{SUB})\\,{SUB - 1})',setpts=N/{FPS}/TB", "-r", str(FPS),
                    "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv444p", str(HERE / f"out/chunk_{k}.mp4")], check=True)
    shutil.rmtree(sub, ignore_errors=True)
    print("chunk", k, "done")


def concat(n):
    lst = HERE / "out/chunks.txt"
    lst.write_text("".join(f"file 'chunk_{k}.mp4'\n" for k in range(n)))
    subprocess.run([FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(HERE / "out/video.mp4")], check=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries", "stream=nb_read_frames",
                        "-of", "csv=p=0", str(HERE / "out/video.mp4")], capture_output=True, text=True)
    print("video frames:", r.stdout.strip(), "expected", int(round(T * FPS)))


def pops(path=None):
    path = path or HERE / "out/video.mp4"
    raw = subprocess.run([FF, "-v", "quiet", "-i", str(path), "-vf", "scale=180:180,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 180, 180).astype(np.float32)
    d = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    hits = [(i + 1, round((i + 1) / FPS, 3), round(float(d[i]), 2)) for i in range(1, len(d) - 1)
            if d[i] > 3 * max(d[i - 1], d[i + 1], 0.3) and d[i] > 2.0]
    print("pops:", len(hits)); [print("  frame", *h) for h in hits]
    fl = [(n, round(n / FPS, 3)) for n in range(1, len(fr) - 1)
          if min(d[n - 1], d[n]) > 2.0 and np.abs(fr[n + 1] - fr[n - 1]).mean() < 0.35 * min(d[n - 1], d[n])]
    print("one-frame flashes:", len(fl)); [print("  frame", *f) for f in fl]


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "probe": asyncio.run(probe([float(x) for x in sys.argv[2:]]))
    elif cmd == "chunk": asyncio.run(chunk(int(sys.argv[2]), int(sys.argv[3])))
    elif cmd == "concat": concat(int(sys.argv[2]))
    elif cmd == "pops": pops()
