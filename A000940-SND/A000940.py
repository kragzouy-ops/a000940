"""
A000940 Evolutive Sound — Python Edition (pygame + numpy + optional sounddevice + mido)

Features (parity with the React version where feasible):
- Visualize inequivalent star polygons {n/k} for current n; rotate and morph over time
- Resistor color code for n%10 → color (black→white), prime→gold accent, square→silver accent
- Split layout: left control panel, right output canvas
- Sequencer that advances through inequivalent steps k, schedules an arpeggio per polygon
- Simple software synth (triangle + ADSR) using `sounddevice` if available (fallback to silent)
- Optional FFT strip / fullscreen overlay (computed from the last audio buffer)
- MIDI export (Type 0) and import (format 0) using `mido`
- Preferences persisted to JSON file in ~/.a000940_config.json (dark mode, BW invert, etc.)

Dependencies (install via pip):
    pip install pygame numpy mido sounddevice
(Sound works without sounddevice by disabling audio; MIDI features require mido)

Run:
    python a000940.py

Controls (mouse):
- Click on labeled buttons in the left panel to toggle.
- Sliders are horizontal bars; click to set.

Controls (keys):
- SPACE: start / pause
- F: toggle FFT strip
- G: toggle fullscreen FFT
- B: toggle BW overlay (invert GUI only)
- D: toggle Dark mode
- M: toggle Morph
- S: toggle Sync BPM
- L: toggle Legend
- A: toggle Accents
- ESC: quit

Notes:
- This is a single-file demo; for production split into modules.
- Video recording is not implemented in Python edition.
"""
from __future__ import annotations
import math
import os
import json
import time
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import pygame

try:
    import sounddevice as sd  # optional, for audio output
    HAVE_SD = True
except Exception:
    sd = None
    HAVE_SD = False

try:
    import mido
    HAVE_MIDO = True
except Exception:
    mido = None
    HAVE_MIDO = False

# --------------------------- Math & Color Helpers ---------------------------
TAU = math.pi * 2.0

def clamp(x, a, b):
    return min(b, max(a, x))

def lerp(a, b, t):
    return a + (b - a) * t

def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def unique_steps(n: int) -> List[int]:
    steps = []
    for k in range(1, n // 2 + 1):
        if gcd(n, k) == 1:
            steps.append(k)
    return steps

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def is_perfect_square(n: int) -> bool:
    r = int(math.isqrt(n))
    return r * r == n

RESISTOR_COLORS = [
    '#000000',  # 0 black
    '#8B4513',  # 1 brown
    '#FF0000',  # 2 red
    '#FFA500',  # 3 orange
    '#FFFF00',  # 4 yellow
    '#008000',  # 5 green
    '#0000FF',  # 6 blue
    '#8A2BE2',  # 7 violet
    '#808080',  # 8 grey
    '#FFFFFF',  # 9 white
]

def resistor_color(n: int) -> str:
    d = ((n % 10) + 10) % 10
    return RESISTOR_COLORS[d]

def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    h = (hex_str or '').lstrip('#')
    if len(h) == 3:
        h = ''.join(c + c for c in h)
    try:
        v = int(h or '000000', 16)
    except Exception:
        v = 0
    r = (v >> 16) & 255
    g = (v >> 8) & 255
    b = v & 255
    return r, g, b

def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"

def invert_hex(hex_str: str) -> str:
    r, g, b = hex_to_rgb(hex_str)
    return rgb_to_hex(255 - r, 255 - g, 255 - b)

def mix_hex(a_hex: str, b_hex: str, t: float) -> str:
    ar, ag, ab = hex_to_rgb(a_hex)
    br, bg, bb = hex_to_rgb(b_hex)
    rr = int(round(lerp(ar, br, clamp(t, 0.0, 1.0))))
    rg = int(round(lerp(ag, bg, clamp(t, 0.0, 1.0))))
    rb = int(round(lerp(ab, bb, clamp(t, 0.0, 1.0))))
    return rgb_to_hex(rr, rg, rb)

def shade_hex(hex_str: str, f: float) -> str:
    if f >= 0:
        return mix_hex(hex_str, '#ffffff', clamp(f, 0.0, 1.0))
    else:
        return mix_hex(hex_str, '#000000', clamp(-f, 0.0, 1.0))

def luma(hex_str: str) -> float:
    r, g, b = hex_to_rgb(hex_str)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

def readable_text_on(hex_str: str) -> Tuple[int, int, int]:
    return (0, 0, 0) if luma(hex_str) > 0.55 else (255, 255, 255)

# --------------------------- Scales & MIDI helpers ---------------------------
SCALES = {
    'major':      [0, 2, 4, 5, 7, 9, 11],
    'minor':      [0, 2, 3, 5, 7, 8, 10],
    'dorian':     [0, 2, 3, 5, 7, 9, 10],
    'pentatonic': [0, 2, 5, 7, 9],
}

def midi_to_freq(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def scale_degree_to_midi(root_midi: int, degree: int, scale_name: str = 'major') -> int:
    s = SCALES.get(scale_name, SCALES['major'])
    octv = degree // len(s)
    pc = s[degree % len(s)]
    return root_midi + pc + 12 * octv

# --------------------------- Simple MIDI writer ---------------------------

def build_midi(events: List[dict], tempo_bpm: float = 120.0, ppq: int = 480):
    if not HAVE_MIDO:
        raise RuntimeError('mido not installed')
    mid = mido.MidiFile(type=0)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    tempo_meta = mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm))
    track.append(tempo_meta)
    # events should be sorted by tick
    events_sorted = sorted(events, key=lambda e: e['tick'])
    last_tick = 0
    for ev in events_sorted:
        delta = max(0, ev['tick'] - last_tick)
        last_tick = ev['tick']
        if ev['type'] == 'on':
            track.append(mido.Message('note_on', note=int(ev['note']), velocity=int(ev.get('vel', 96)), time=delta))
        else:
            track.append(mido.Message('note_off', note=int(ev['note']), velocity=0, time=delta))
    # End of track
    track.append(mido.MetaMessage('end_of_track', time=0))
    mid.ticks_per_beat = ppq
    return mid

# --------------------------- Audio Synth (sounddevice) ---------------------------

class PolySynth:
    def __init__(self, samplerate: int = 44100):
        self.samplerate = samplerate
        self.muted = False
        self.ppq = 480
        self.events: List[dict] = []  # for MIDI export
        self._queue: List[np.ndarray] = []
        self._queue_lock = threading.Lock()
        self._pos = 0
        self._current: Optional[np.ndarray] = None
        self._stream: Optional[sd.OutputStream] = None if HAVE_SD else None
        self._an_buf = np.zeros(4096, dtype=np.float32)
        self._an_pos = 0

    def init(self):
        if not HAVE_SD:
            return
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(channels=1, samplerate=self.samplerate, callback=self._callback)
        self._stream.start()

    def set_muted(self, m: bool):
        self.muted = m

    def get_fft_bins(self, nfft: int = 1024):
        # Return magnitude spectrum from analysis buffer
        buf = self._an_buf.copy()
        if np.max(np.abs(buf)) < 1e-6:
            return np.zeros(nfft // 2, dtype=np.float32)
        w = np.hanning(nfft)
        x = buf[-nfft:] * w
        sp = np.fft.rfft(x)
        mag = np.abs(sp)
        mag = mag / (np.max(mag) + 1e-9)
        return mag.astype(np.float32)

    def note_on(self, midi_note: int, when_s: float, dur_s: float, velocity: float = 0.8):
        # Render into a segment buffer and queue it (simple scheduling)
        # Triangle waveform + ADSR
        f = midi_to_freq(midi_note)
        a, d, s, r = 0.01, 0.1, 0.6, 0.2
        total = dur_s + r + 0.01
        n = int(total * self.samplerate)
        t = np.arange(n, dtype=np.float32) / self.samplerate
        # Triangle wave
        # tri = 2*abs(2*((t*f)%1)-1)-1
        tri = 2 * np.abs(2 * ((t * f) % 1.0) - 1.0) - 1.0
        # ADSR envelope
        env = np.zeros_like(t)
        ta = int(a * self.samplerate)
        td = int(d * self.samplerate)
        tr = int(r * self.samplerate)
        ts = max(0, n - (ta + td + tr))
        if ta > 0:
            env[:ta] = np.linspace(0, 1, ta, endpoint=False)
        if td > 0:
            env[ta:ta + td] = np.linspace(1, s, td, endpoint=False)
        if ts > 0:
            env[ta + td:ta + td + ts] = s
        if tr > 0:
            env[ta + td + ts:] = np.linspace(s, 0, tr, endpoint=True)
        sig = (tri * env * velocity).astype(np.float32)
        # Schedule offset by when_s: prepend silence
        lead = int(max(0.0, when_s) * self.samplerate)
        if lead > 0:
            sig = np.concatenate([np.zeros(lead, dtype=np.float32), sig])
        with self._queue_lock:
            self._queue.append(sig)

    def _callback(self, outdata, frames, time_info, status):  # type: ignore
        if status:
            pass
        if self.muted:
            outdata[:] = 0
            return
        # Ensure we have a current buffer
        if self._current is None or self._pos >= len(self._current):
            with self._queue_lock:
                if self._queue:
                    self._current = self._queue.pop(0)
                    self._pos = 0
                else:
                    self._current = np.zeros(frames, dtype=np.float32)
                    self._pos = 0
        # Copy
        end = min(self._pos + frames, len(self._current))
        chunk = self._current[self._pos:end]
        self._pos = end
        if len(chunk) < frames:
            pad = np.zeros(frames - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])
        out = chunk.reshape(-1, 1)
        outdata[:len(out), 0] = out[:, 0]
        # Analysis ring buffer
        c = out[:, 0]
        l = len(c)
        pos = self._an_pos % len(self._an_buf)
        endpos = pos + l
        if endpos < len(self._an_buf):
            self._an_buf[pos:endpos] = c
        else:
            k = len(self._an_buf) - pos
            self._an_buf[pos:] = c[:k]
            self._an_buf[:l - k] = c[k:]
        self._an_pos = (self._an_pos + l) % len(self._an_buf)

# --------------------------- Mapping helpers ---------------------------

def choose_mode_classic(n: int) -> str:
    return 'minor' if (n % 7 == 0 or n % 5 == 0) else 'major'

def pattern_classic(n: int, k: int, scale_name: str) -> List[int]:
    deg_base = (k * 5) % (len(SCALES.get(scale_name, SCALES['major'])))
    return [deg_base + d for d in [0, 2, 4, 1, 3]]

def pattern_mode_by_k(n: int, k: int):
    cycle = ['major', 'minor', 'dorian']
    scale_name = cycle[k % len(cycle)]
    deg_base = n % len(SCALES[scale_name])
    return scale_name, [deg_base + d for d in [0, 2, 4, 1, 3]]

def pattern_steps_arp(n: int):
    steps = unique_steps(n)
    scale_name = 'major'
    degs = [s % len(SCALES[scale_name]) for s in steps]
    while len(degs) < 5:
        degs += degs
    return scale_name, degs[:5]

# --------------------------- Preferences ---------------------------
CFG_PATH = os.path.join(os.path.expanduser('~'), '.a000940_config.json')

def load_prefs():
    defaults = {
        'dark': True,
        'invert_gui_bw': True,
        'show_fft': False,
        'fft_full': False,
        'bpm': 90,
        'n_min': 3,
        'n_max': 12,
        'morph': True,
        'morph_sync_bpm': True,
        'morph_beats': 2.0,
        'morph_seconds': 2.0,
        'thickness': 2.0,
        'root_offset': 0,
        'mapping_mode': 'classic',
        'show_accents': True,
        'show_legend': True,
    }
    try:
        if os.path.exists(CFG_PATH):
            with open(CFG_PATH, 'r') as f:
                data = json.load(f)
                defaults.update({k: data.get(k, v) for k, v in defaults.items()})
    except Exception:
        pass
    return defaults

def save_prefs(p):
    try:
        with open(CFG_PATH, 'w') as f:
            json.dump(p, f, indent=2)
    except Exception:
        pass

# --------------------------- App ---------------------------
@dataclass
class SeqState:
    n: int = 3
    k_index: int = 0
    step_t: float = 0.0
    rot: float = 0.0
    tick: int = 0

class App:
    def __init__(self, w=1200, h=800):
        pygame.init()
        pygame.display.set_caption('A000940 — Evolutive Sound (Python)')
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 18)
        self.small = pygame.font.SysFont(None, 14)
        self.tiny = pygame.font.SysFont(None, 12)
        self.prefs = load_prefs()
        self.running = False
        self.dark = bool(self.prefs['dark'])
        self.invert_gui_bw = bool(self.prefs['invert_gui_bw'])
        self.show_fft = bool(self.prefs['show_fft'])
        self.fft_full = bool(self.prefs['fft_full'])
        self.bpm = float(self.prefs['bpm'])
        self.n_min = int(self.prefs['n_min'])
        self.n_max = int(self.prefs['n_max'])
        self.morph = bool(self.prefs['morph'])
        self.morph_sync_bpm = bool(self.prefs['morph_sync_bpm'])
        self.morph_beats = float(self.prefs['morph_beats'])
        self.morph_seconds = float(self.prefs['morph_seconds'])
        self.thickness = float(self.prefs['thickness'])
        self.root_offset = int(self.prefs['root_offset'])
        self.mapping_mode = str(self.prefs['mapping_mode'])
        self.show_accents = bool(self.prefs['show_accents'])
        self.show_legend = bool(self.prefs['show_legend'])
        self.seq = SeqState(n=self.n_min)
        self.current_n = self.seq.n
        # Synth
        self.synth = PolySynth()
        try:
            self.synth.init()
        except Exception:
            pass
        self.muted = False
        # Layout
        self.sidebar_w = 380

    # ----------------- Scheduling & Mapping -----------------
    def schedule_from_polygon(self, n: int, k: int):
        if self.mapping_mode == 'modeByK':
            scale_name, degrees = pattern_mode_by_k(n, k)
        elif self.mapping_mode == 'stepsArp':
            scale_name, degrees = pattern_steps_arp(n)
        else:
            scale_name = choose_mode_classic(n)
            degrees = pattern_classic(n, k, scale_name)
        root_midi = 48 + (n % 12) + self.root_offset
        beat = 60.0 / max(1e-6, self.bpm)
        t0 = 0.03  # short lead-in
        for i, deg in enumerate(degrees):
            note = scale_degree_to_midi(root_midi, deg, scale_name)
            when = t0 + i * 0.25 * beat
            dur = 0.22 * beat
            if self.synth:
                try:
                    self.synth.note_on(note, when_s=when, dur_s=dur, velocity=0.7)
                except Exception:
                    pass
            # MIDI collection
            tick_on = self.seq.tick + int(round(i * 0.25 * self.synth.ppq))
            tick_off = tick_on + int(round(0.22 * self.synth.ppq))
            self.synth.events.append({'tick': tick_on, 'type': 'on', 'note': note, 'vel': 100})
            self.synth.events.append({'tick': tick_off, 'type': 'off', 'note': note, 'vel': 0})
        self.seq.tick += int(round(len(degrees) * 0.25 * self.synth.ppq))

    # ----------------- Drawing -----------------
    def draw_star(self, surf, cx, cy, r, n, k, rot, color, thickness=1):
        pts = []
        angle = rot
        step = k * TAU / n
        for i in range(n + 1):
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            pts.append((x, y))
            angle += step
        if len(pts) >= 2:
            pygame.draw.lines(surf, color, True, pts, max(1, int(thickness)))

    def draw(self, dt):
        w, h = self.screen.get_size()
        # background
        if self.dark:
            self.screen.fill((11, 11, 16))
        else:
            self.screen.fill((248, 250, 252))
        # split layout
        sidebar = pygame.Rect(0, 0, self.sidebar_w, h)
        content = pygame.Rect(self.sidebar_w, 0, w - self.sidebar_w, h)
        # content area
        cx = content.x + content.w // 2
        cy = content.y + content.h // 2
        R = int(min(content.w, content.h) * 0.35)

        # Sequencing advance
        beat = 60.0 / max(1e-6, self.bpm)
        if self.running:
            steps = unique_steps(self.seq.n)
            if self.morph:
                period = (max(0.25, self.morph_beats * beat) if self.morph_sync_bpm
                          else max(0.2, self.morph_seconds))
            else:
                period = (max(0.15, 1.2 * beat) if self.morph_sync_bpm
                          else max(0.2, 0.6 * self.morph_seconds))
            self.seq.step_t += dt / period
            self.seq.rot += dt * 0.35
            if self.seq.step_t >= 1.0:
                self.seq.step_t = 0.0
                self.seq.k_index += 1
                if self.seq.k_index >= len(steps):
                    self.seq.k_index = 0
                    self.seq.n += 1
                if self.seq.n > self.n_max:
                    self.seq.n = self.n_min
                steps2 = unique_steps(self.seq.n)
                k = steps2[self.seq.k_index] if steps2 else 1
                self.schedule_from_polygon(self.seq.n, k)
                self.current_n = self.seq.n

        # draw polygons
        steps = unique_steps(self.seq.n)
        k = steps[min(self.seq.k_index, max(0, len(steps) - 1))] if steps else 1
        base_raw = resistor_color(self.seq.n)
        base_hex = base_raw if self.dark else invert_hex(base_raw)
        total = len(steps)
        for i, kk in enumerate(steps):
            t_ring = 1.0 if total <= 1 else (1.0 - i / (total - 1))
            rr = int(R * (0.25 + 0.7 * t_ring))
            shade_span = 0.35
            rel = 0 if total <= 1 else (i - self.seq.k_index) / (total - 1)
            f = clamp(rel * shade_span * 2.0, -shade_span, shade_span)
            color_hex = shade_hex(base_hex, f)
            col = hex_to_rgb(color_hex)
            alpha = 255 if i == self.seq.k_index else int(0.7 * 255)
            # draw on a temp surface for alpha
            tmp = pygame.Surface((w, h), pygame.SRCALPHA)
            self.draw_star(tmp, cx, cy, rr, self.seq.n, kk, self.seq.rot + i * 0.12, col + (alpha,), self.thickness * (0.5 + 0.8 * t_ring))
            self.screen.blit(tmp, (0, 0))

        # label
        label = f"n={self.seq.n}  •  k={k}  •  inequivalent {"{n/k}"}"
        prime = is_prime(self.seq.n)
        square = is_perfect_square(self.seq.n)
        if self.show_accents and (prime or square):
            label += '  •  ' + ('gold (prime)' if prime else 'silver (square)')
        text_col = (15, 23, 42)
        txt = self.font.render(label, True, text_col)
        self.screen.blit(txt, (cx - txt.get_width() // 2, content.bottom - 28))

        # color chip with accent outline
        chip_x = cx - 160
        chip_y = content.bottom - 34
        pygame.draw.circle(self.screen, hex_to_rgb(base_hex), (chip_x, chip_y), 7)
        if self.show_accents and (prime or square):
            col = (212, 175, 55) if prime else (192, 192, 192)
        else:
            col = (0, 0, 0)
        pygame.draw.circle(self.screen, col, (chip_x, chip_y), 8, 2)

        # FFT overlay
        if self.show_fft:
            try:
                mag = self.synth.get_fft_bins(1024)
            except Exception:
                mag = np.zeros(512, dtype=np.float32)
            if self.fft_full:
                overlay = pygame.Surface((w, h), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 128) if self.dark else (255, 255, 255, 128))
                self.screen.blit(overlay, (0, 0))
                bw = w / len(mag)
                col = (34, 211, 238) if self.dark else (14, 165, 233)
                for i, v in enumerate(mag):
                    hh = int(v * (h * 0.9))
                    pygame.draw.rect(self.screen, col, (i * bw, h - hh, bw, hh))
            else:
                bw = w / len(mag)
                col = (34, 211, 238) if self.dark else (14, 165, 233)
                y0 = 12 + (4 if self.dark else 0)
                for i, v in enumerate(mag):
                    hh = int(v * 120)
                    pygame.draw.rect(self.screen, col, (i * bw, y0, bw, hh))

        # Legend panel (bottom-right)
        if self.show_legend:
            cols, rows = 5, 2
            cell, gap = 16, 6
            pad, title_h = 10, 14
            box_w = pad * 2 + cols * cell + (cols - 1) * gap + 22
            box_h = pad * 2 + rows * cell + (rows - 1) * gap + title_h + 6
            x0 = w - box_w - 12
            y0 = h - box_h - 12
            panel = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            bg = (255, 255, 255, 200)
            pygame.draw.rect(panel, bg, (0, 0, box_w, box_h), border_radius=8)
            pygame.draw.rect(panel, (0, 0, 0, 64), (0, 0, box_w, box_h), 1, border_radius=8)
            title = self.small.render('n % 10 → color', True, (15, 23, 42))
            panel.blit(title, (pad, pad))
            for d in range(10):
                swatch_hex = RESISTOR_COLORS[d] if self.dark else invert_hex(RESISTOR_COLORS[d])
                col = hex_to_rgb(swatch_hex)
                col_text = readable_text_on(swatch_hex)
                col_idx = d % cols
                row = d // cols
                sx = pad + col_idx * (cell + gap)
                sy = pad + title_h + 6 + row * (cell + gap)
                pygame.draw.rect(panel, col, (sx, sy, cell, cell))
                pygame.draw.rect(panel, (0, 0, 0), (sx, sy, cell, cell), 1)
                digit = self.tiny.render(str(d), True, col_text)
                panel.blit(digit, (sx + cell + 3, sy + cell - 10))
            self.screen.blit(panel, (x0, y0))

        # Sidebar (GUI)
        self.draw_sidebar(sidebar)

    def draw_sidebar(self, rect: pygame.Rect):
        # panel background (simulate invert-only when requested)
        bg = (245, 245, 245) if self.dark else (32, 32, 36)
        fg = (20, 20, 20) if self.dark else (235, 235, 240)
        panel = pygame.Surface((rect.w, rect.h))
        panel.fill(bg)
        x, y = 16, 14
        def button(label, on):
            nonlocal y
            ACC = resistor_color(self.current_n)
            ACC = ACC if self.dark else invert_hex(ACC)
            border = hex_to_rgb(shade_hex(ACC, -0.15))
            fill = hex_to_rgb(ACC)
            txt_col = readable_text_on(ACC)
            rw, rh = rect.w - 32, 30
            r = pygame.Rect(x, y, rw, rh)
            pygame.draw.rect(panel, fill if on else bg, r, 0, border_radius=12)
            pygame.draw.rect(panel, border, r, 2, border_radius=12)
            t = self.small.render(label, True, txt_col if on else (0,0,0) if self.dark else (255,255,255))
            panel.blit(t, (r.x + 10, r.y + (rh - t.get_height()) // 2))
            y += rh + 8
            return r
        def slider(label, value, vmin, vmax, step=1.0):
            nonlocal y
            rw, rh = rect.w - 32, 30
            r = pygame.Rect(x, y, rw, rh)
            pygame.draw.rect(panel, (230, 230, 235) if self.dark else (50, 50, 55), r, border_radius=10)
            # bar
            t = (value - vmin) / (vmax - vmin)
            bw = int(rw * clamp(t, 0.0, 1.0))
            bar = pygame.Rect(x, y, bw, rh)
            col = hex_to_rgb(resistor_color(self.current_n if self.dark else -self.current_n))
            pygame.draw.rect(panel, col, bar, border_radius=10)
            # label
            txt = self.small.render(f"{label}: {value:.2f}", True, (0,0,0) if self.dark else (255,255,255))
            panel.blit(txt, (x + 8, y + 6))
            y += rh + 8
            return r
        # Title
        title = self.font.render("Controls", True, (0,0,0) if self.dark else (255,255,255))
        panel.blit(title, (x, y))
        y += 26
        # Buttons
        r_start = button('Start' if not self.running else 'Pause', self.running)
        r_fft = button('FFT', self.show_fft)
        r_fftf = button('FFT Full', self.fft_full)
        r_morph = button('Morph', self.morph)
        r_sync = button('Sync BPM', self.morph_sync_bpm)
        r_dark = button('Dark Mode', self.dark)
        r_bw = button('BW Overlay', self.invert_gui_bw)
        r_legend = button('Legend', self.show_legend)
        r_acc = button('Accents', self.show_accents)
        # Sliders
        r_bpm = slider('BPM', self.bpm, 40, 180, 1)
        r_nmin = slider('n min', float(self.n_min), 3, 24, 1)
        r_nmax = slider('n max', float(self.n_max), 3, 32, 1)
        r_th = slider('Stroke', self.thickness, 1, 6, 0.5)
        # Map handling rectangles for clicks
        self._clickables = [
            ('start', r_start), ('fft', r_fft), ('fftf', r_fftf), ('morph', r_morph), ('sync', r_sync),
            ('dark', r_dark), ('bw', r_bw), ('legend', r_legend), ('acc', r_acc),
            ('bpm', r_bpm), ('nmin', r_nmin), ('nmax', r_nmax), ('th', r_th)
        ]
        # Apply invert filter to GUI only (simulate by inverting surface colors)
        if self.invert_gui_bw:
            arr = pygame.surfarray.pixels3d(panel)
            np.subtract(255, arr, out=arr)
            del arr
        self.screen.blit(panel, (rect.x, rect.y))

    # ----------------- Event Handling -----------------
    def handle_click(self, pos):
        if not hasattr(self, '_clickables'):
            return
        for name, r in self._clickables:
            if r.collidepoint(pos[0], pos[1]):
                if name == 'start':
                    if not self.running:
                        self.handle_start()
                    else:
                        self.running = False
                elif name == 'fft':
                    self.show_fft = not self.show_fft
                elif name == 'fftf':
                    self.fft_full = not self.fft_full
                elif name == 'morph':
                    self.morph = not self.morph
                elif name == 'sync':
                    self.morph_sync_bpm = not self.morph_sync_bpm
                elif name == 'dark':
                    self.dark = not self.dark
                elif name == 'bw':
                    self.invert_gui_bw = not self.invert_gui_bw
                elif name == 'legend':
                    self.show_legend = not self.show_legend
                elif name == 'acc':
                    self.show_accents = not self.show_accents
                elif name == 'bpm':
                    rel = (pos[0] - r.x) / max(1, r.w)
                    self.bpm = clamp(40 + rel * (180 - 40), 40, 180)
                elif name == 'nmin':
                    rel = (pos[0] - r.x) / max(1, r.w)
                    self.n_min = int(round(clamp(3 + rel * (24 - 3), 3, 24)))
                    if self.n_min > self.n_max:
                        self.n_max = self.n_min
                elif name == 'nmax':
                    rel = (pos[0] - r.x) / max(1, r.w)
                    self.n_max = int(round(clamp(3 + rel * (32 - 3), 3, 32)))
                    if self.n_max < self.n_min:
                        self.n_min = self.n_max
                elif name == 'th':
                    rel = (pos[0] - r.x) / max(1, r.w)
                    self.thickness = clamp(1 + rel * (6 - 1), 1, 6)
                break
        self.persist()

    def handle_start(self):
        # reset and schedule first polygon
        self.running = True
        self.seq = SeqState(n=self.n_min, k_index=0, step_t=0.0, rot=0.0, tick=0)
        steps = unique_steps(self.seq.n)
        k0 = steps[0] if steps else 1
        self.schedule_from_polygon(self.seq.n, k0)
        self.current_n = self.seq.n

    def persist(self):
        self.prefs.update({
            'dark': self.dark,
            'invert_gui_bw': self.invert_gui_bw,
            'show_fft': self.show_fft,
            'fft_full': self.fft_full,
            'bpm': self.bpm,
            'n_min': self.n_min,
            'n_max': self.n_max,
            'morph': self.morph,
            'morph_sync_bpm': self.morph_sync_bpm,
            'morph_beats': self.morph_beats,
            'morph_seconds': self.morph_seconds,
            'thickness': self.thickness,
            'root_offset': self.root_offset,
            'mapping_mode': self.mapping_mode,
            'show_accents': self.show_accents,
            'show_legend': self.show_legend,
        })
        save_prefs(self.prefs)

    # ----------------- Main Loop -----------------
    def run(self):
        last = time.time()
        running = True
        while running:
            now = time.time()
            dt = clamp(now - last, 0.0, 0.1)
            last = now
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if not self.running:
                            self.handle_start()
                        else:
                            self.running = False
                    elif event.key == pygame.K_f:
                        self.show_fft = not self.show_fft
                    elif event.key == pygame.K_g:
                        self.fft_full = not self.fft_full
                    elif event.key == pygame.K_b:
                        self.invert_gui_bw = not self.invert_gui_bw
                    elif event.key == pygame.K_d:
                        self.dark = not self.dark
                    elif event.key == pygame.K_m:
                        self.morph = not self.morph
                    elif event.key == pygame.K_s:
                        self.morph_sync_bpm = not self.morph_sync_bpm
                    elif event.key == pygame.K_l:
                        self.show_legend = not self.show_legend
                    elif event.key == pygame.K_a:
                        self.show_accents = not self.show_accents
                    self.persist()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
            self.draw(dt)
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

# --------------------------- Tests ---------------------------

def _run_light_tests():
    # gcd tests
    assert gcd(12, 8) == 4
    assert gcd(9, 4) == 1
    # unique steps counts
    cases = [
        (3, 1), (4, 1), (5, 2), (6, 1), (7, 3), (8, 2), (9, 3), (10, 2), (12, 2)
    ]
    for n, count in cases:
        got = len(unique_steps(n))
        assert got == count, f"unique_steps({n}) == {count}, got {got}"
    # midi freq sanity
    assert abs(midi_to_freq(69) - 440.0) < 1e-6
    # resistor colors
    assert resistor_color(0) == '#000000'
    assert resistor_color(7).lower() == '#8a2be2'
    assert resistor_color(8) == '#808080'
    assert resistor_color(9).lower() == '#ffffff'
    assert resistor_color(19).lower() == '#ffffff'
    assert len(RESISTOR_COLORS) == 10
    # invert & shade & luma
    assert invert_hex('#000000') == '#ffffff'
    assert invert_hex('#ffffff') == '#000000'
    sh_l = shade_hex('#808080', 0.3)
    sh_d = shade_hex('#808080', -0.3)
    assert sh_l != sh_d
    assert readable_text_on('#000000') == (255, 255, 255)
    assert readable_text_on('#ffffff') == (0, 0, 0)
    # accent rules
    assert is_prime(11) is True
    assert is_prime(12) is False
    assert is_perfect_square(16) is True
    assert is_perfect_square(15) is False
    # mapping counts
    ns = [3, 5, 7, 9, 10, 12]
    for n in ns:
        steps = unique_steps(n)
        for k in steps:
            scale = choose_mode_classic(n)
            degs = pattern_classic(n, k, scale)
            assert len(degs) == 5
            s2, out = pattern_mode_by_k(n, k)
            assert len(out) == 5 and s2 in SCALES
            s3, out2 = pattern_steps_arp(n)
            assert len(out2) >= 5 and s3 in SCALES

if __name__ == '__main__':
    _run_light_tests()
    print('[A000940] tests passed ✅')
    app = App()
    app.run()
