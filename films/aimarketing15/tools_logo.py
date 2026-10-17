# Cut the AIMarketing logo out of the reference end card (light background) into ring + wordmark PNGs.
import numpy as np
from PIL import Image
a = np.asarray(Image.open('assets/logo_raw.png').convert('RGB')).astype(float)
H, W, _ = a.shape
# background estimated per row/col from the frame border (gentle gradient)
bg = np.zeros_like(a)
left, right = a[:, :6].mean(1), a[:, -6:].mean(1)
xs = np.linspace(0, 1, W)[None, :, None]
bg = left[:, None, :] * (1 - xs) + right[:, None, :] * xs
alpha = np.clip((bg[..., 0] - a[..., 0]) / (bg[..., 0] - 26), 0, 1)
alpha[alpha < 0.04] = 0
col = (a - (1 - alpha[..., None]) * bg) / np.maximum(alpha[..., None], 1e-3)
col = np.clip(col, 0, 255)
green = (col[..., 1] > col[..., 2] + 30)
out = np.dstack([col, alpha * 255]).astype(np.uint8)
ring = out.copy(); ring[~green, 3] = 0
word = out.copy(); word[green, 3] = 0
for name, im in [('logo', out), ('ring', ring), ('word', word)]:
    Image.fromarray(im, 'RGBA').save(f'assets/{name}.png')
ys, xs2 = np.nonzero(green & (alpha > .5))
print('ring bbox', xs2.min(), ys.min(), xs2.max(), ys.max(), 'size', W, H)
