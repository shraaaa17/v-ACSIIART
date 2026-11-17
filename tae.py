"""
Random letters -> ASCII portrait, played LIVE in a window (tkinter, same as turtle).

Install:  pip install numpy opencv-python pillow
Run:      python ascii_window.py
Keys:     R or click = replay      Esc = close
"""
import time
import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk

# ------------------------- SETTINGS (change these) -------------------------
SRC  = 'photo.jpg'                 # your input image
FONT = 'DejaVuSansMono-Bold.ttf'   # any monospace .ttf (see note at bottom)
CW, CH = 5, 9                      # one character cell in pixels
COLS, ROWS = 256, 97               # characters across / down  -> window = COLS*CW x ROWS*CH
                                   # (bigger = more detail but slower; 384 x 146 = full HD)
RAMP = " :;irsXA253hMHGS#988"      # light -> dense
SUBJECT_RECT = (215, 80, 295, 334) # (x, y, w, h) box around the person in YOUR photo
REVEAL_SECONDS = 6.8               # how long the reveal takes
# ---------------------------------------------------------------------------

W, H = COLS * CW, ROWS * CH
rng = np.random.default_rng(7)

# ---------- 1. Atlas: a tiny picture of every character ----------
font = None
for name in (FONT, 'consolab.ttf', 'cour.ttf', 'Menlo.ttc',
             '/System/Library/Fonts/Menlo.ttc',
             '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'):
    try:
        font = ImageFont.truetype(name, 8)      # try each font until one is found
        break
    except OSError:
        continue
if font is None:
    font = ImageFont.load_default()
chars = [chr(c) for c in range(33, 127)]
atlas = np.zeros((len(chars) + 1, CH, CW), np.float32)      # index 0 = empty cell
for i, ch in enumerate(chars, 1):
    im = Image.new('L', (CW, CH), 0)
    d = ImageDraw.Draw(im)
    bb = d.textbbox((0, 0), ch, font=font)
    d.text(((CW - (bb[2] - bb[0])) // 2 - bb[0], (CH - (bb[3] - bb[1])) // 2 - bb[1]),
           ch, fill=255, font=font)
    atlas[i] = np.asarray(im, np.float32) / 255.
cidx = {c: i for i, c in enumerate(chars, 1)}
rand_pool = np.array([cidx[c] for c in
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#$%&*+=<>?/\\|{}[]'])

# ---------- 2. Photo -> grid of characters ----------
img = cv2.imread(SRC)
if img is None:
    raise SystemExit(f'Could not open "{SRC}". Put your photo next to this script.')

mask = np.zeros(img.shape[:2], np.uint8)                     # cut the person out
cv2.grabCut(img, mask, SUBJECT_RECT, np.zeros((1, 65)), np.zeros((1, 65)),
            8, cv2.GC_INIT_WITH_RECT)
m = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
m = cv2.GaussianBlur(m, (0, 0), 3).astype(np.float32) / 255.

g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sub = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(3, 3)).apply(g0).astype(np.float32)
sl, sh = np.percentile(sub[m > 0.5], [2, 99])
sub = 0.05 + 0.95 * np.clip((sub - sl) / (sh - sl), 0, 1) ** 1.3   # person: strong contrast
bg = (g0.astype(np.float32) / 255.) ** 1.1 * 0.30                  # background: dim
rim = cv2.GaussianBlur(np.abs(cv2.dilate(m, np.ones((5, 5))) -
                              cv2.erode(m, np.ones((5, 5)))), (0, 0), 1.2)
comb = np.clip(sub * m + bg * (1 - m) + 0.55 * rim, 0, 1)

g = cv2.resize(comb * 255, (COLS, ROWS), interpolation=cv2.INTER_AREA).astype(np.float32)
g = np.clip(g + 1.1 * (g - cv2.GaussianBlur(g, (0, 0), 1.2)), 0, 255)   # sharpen
lo, hi = np.percentile(g, [1, 99.5])
b = np.clip((g - lo) / (hi - lo), 0, 1)

lv = np.minimum((b * len(RAMP)).astype(int), len(RAMP) - 1)
target = np.array([cidx.get(RAMP[l], 0) for l in lv.ravel()]).reshape(ROWS, COLS)
target[lv == 0] = 0
tint_strength = 0.30 + 0.70 * b

# ---------- 3. Timing: when does each cell lock in? ----------
yy, xx = np.mgrid[0:ROWS, 0:COLS]
dist = np.hypot((xx - COLS / 2) / COLS * 1.45, (yy - ROWS / 2) / ROWS)
t_rev = 1.2 + (REVEAL_SECONDS - 1.8) * dist / dist.max() + rng.random((ROWS, COLS)) * 1.1
phase = rng.integers(0, 2, (ROWS, COLS))
_rand_cache = {'tick': -1, 'idx': None}

def random_letters(t):
    tick = int(t * 15)                       # new random letters 15 times per second
    if tick != _rand_cache['tick']:
        _rand_cache['tick'] = tick
        _rand_cache['idx'] = rand_pool[rng.integers(0, len(rand_pool), (ROWS, COLS))]
    return _rand_cache['idx']

def render(t):
    """Return the picture (H x W x 3, RGB) for time t seconds after start."""
    rnd = random_letters(t)
    fade_in = min(1, t / 0.6)
    dt = t - t_rev                           # >= 0  ->  this cell is revealed
    locked = dt >= 0

    idx = np.where(locked, target, rnd)
    pre = 0.22 + 0.5 * np.clip(1 + dt / 0.9, 0, 1)          # noise brightens before locking
    flash = np.clip(1 - dt / 0.35, 0, 1)                    # white flash when locking
    post = np.where(target == 0, 0.0, tint_strength * (1 + 0.55 * flash))
    shimmer = 1 + 0.04 * np.sin(t * 3 + xx * 0.15 + yy * 0.2)
    flick = np.where(((int(t * 30) + phase) % 2) == 0, 1.0, 0.85)
    inten = np.where(locked, post * shimmer, pre * flick) * fade_in

    empty = locked & (target == 0)                          # empty cells: noise dies fast
    idx = np.where(empty, rnd, idx)
    inten = np.where(empty, np.clip(1 - dt / 0.25, 0, 1) * 0.5 * fade_in, inten)
    inten = np.clip(inten, 0, 1.6)

    v = atlas[idx] * inten[..., None, None]                 # paste all glyphs
    v = v.transpose(0, 2, 1, 3).reshape(H, W)

    small = cv2.resize(v, (W // 2, H // 2), interpolation=cv2.INTER_AREA)   # glow (half-res = fast)
    glow = cv2.GaussianBlur(small, (0, 0), 1.3) * 0.35 + cv2.GaussianBlur(small, (0, 0), 4.0) * 0.45
    glow = cv2.resize(glow, (W, H))
    lum = np.clip(v + glow, 0, 1)
    rgb = np.stack([lum * 0.82, lum * 0.98, lum], -1)       # white with slight cyan tint
    return (rgb * 255).astype(np.uint8)

# ---------- 4. The window ----------
def main():
    root = tk.Tk()
    root.title('ASCII Reveal')
    root.configure(bg='black')
    root.resizable(True, True)
    canvas = tk.Canvas(root, width=W, height=H, bg='black', highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    item = canvas.create_image(0, 0, anchor='center')
    state = {'start': time.perf_counter(), 'photo': None}

    def tick():
        t = time.perf_counter() - state['start']            # real clock -> correct speed
        frame = Image.fromarray(render(t))
        canvas_width = max(canvas.winfo_width(), 1)
        canvas_height = max(canvas.winfo_height(), 1)
        scale = min(canvas_width / W, canvas_height / H)
        display_size = (max(1, int(W * scale)), max(1, int(H * scale)))
        if display_size != (W, H):
            frame = frame.resize(display_size, Image.Resampling.LANCZOS)
        state['photo'] = ImageTk.PhotoImage(frame)
        canvas.coords(item, canvas_width // 2, canvas_height // 2)
        canvas.itemconfig(item, image=state['photo'])       # keep a reference or it vanishes
        root.after(1, tick)

    def replay(event=None):
        state['start'] = time.perf_counter()

    root.bind('r', replay)
    root.bind('R', replay)
    canvas.bind('<Button-1>', replay)
    root.bind('<Escape>', lambda e: root.destroy())
    tick()
    root.mainloop()

if __name__ == '__main__':
    main()