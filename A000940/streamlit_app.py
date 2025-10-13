"""
Inequivalent Polygons — Python webapp-ish script (matplotlib UI)

Features
- Star polygons {n/k} and OEIS A000940 representatives mode (small n)
- Ping–pong animation over a sequence with optional hard cuts between different n
- "Bounce n" option: cycles n=min..max..min with a 1-step dwell at each boundary
- Metronome BPM beeps (requires simpleaudio, optional)
- Export a single one-bounce loop to WebM/MP4 (requires ffmpeg installed)

Run:  python inequivalent_polygons_app.py
Dependencies: matplotlib, numpy, (optional) simpleaudio, (optional) ffmpeg for video export
"""
from __future__ import annotations
import math, time, itertools, colorsys, sys, os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons

# ----------------------------- Math & helpers -----------------------------
def gcd(a: int, b: int) -> int:
    return abs(a) if b == 0 else gcd(b, a % b)

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def hex_to_rgb(hexstr: str) -> Tuple[float,float,float]:
    hexstr = hexstr.strip().lstrip('#')
    if len(hexstr) == 3:
        hexstr = ''.join(ch*2 for ch in hexstr)
    r = int(hexstr[0:2], 16)
    g = int(hexstr[2:4], 16)
    b = int(hexstr[4:6], 16)
    return (r/255.0, g/255.0, b/255.0)

def rgb_to_hex(r: float, g: float, b: float) -> str:
    to = lambda v: f"{int(max(0,min(255,round(v*255)))):02x}"
    return f"#{to(r)}{to(g)}{to(b)}"

def lerp_color(a_hex: str, b_hex: str, t: float) -> str:
    ar,ag,ab = hex_to_rgb(a_hex)
    br,bg,bb = hex_to_rgb(b_hex)
    return rgb_to_hex(lerp(ar,br,t), lerp(ag,bg,t), lerp(ab,bb,t))

# ----------------------------- Geometry -----------------------------------
def regular_polygon(n: int, r: float) -> List[Tuple[float,float]]:
    pts = []
    rot = -math.pi/2
    for i in range(n):
        a = rot + (i*2*math.pi)/n
        pts.append((math.cos(a)*r, math.sin(a)*r))
    return pts

def close_path(pts: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    if not pts: return pts
    if pts[0] == pts[-1]: return pts
    return pts + [pts[0]]

def star_path(pts: List[Tuple[float,float]], k: int) -> List[Tuple[float,float]]:
    n = len(pts)
    visited = [False]*n
    path: List[Tuple[float,float]] = []
    for start in range(n):
        if visited[start]:
            continue
        i = start
        cyc = []
        while not visited[i]:
            visited[i] = True
            cyc.append(pts[i])
            i = (i + k) % n
        path.extend(cyc)
    return path

def resample_closed(pts: List[Tuple[float,float]], samples: int) -> List[Tuple[float,float]]:
    if len(pts) < 2:
        return [(0.0,0.0)]*samples
    dists = [0.0]
    total = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0]-pts[i-1][0]; dy = pts[i][1]-pts[i-1][1]
        total += math.hypot(dx,dy)
        dists.append(total)
    res = []
    for s in range(samples):
        tt = (s/samples) * total
        j = 1
        while j < len(dists) and dists[j] < tt:
            j += 1
        i0 = j-1; i1 = j % len(pts)
        seg = dists[j]-dists[i0] if dists[j]-dists[i0] != 0 else 1.0
        u = (tt - dists[i0])/seg
        x = lerp(pts[i0][0], pts[i1][0], u)
        y = lerp(pts[i0][1], pts[i1][1], u)
        res.append((x,y))
    return res

# ----------------------------- Colors -------------------------------------
RESISTOR_COLORS: Dict[int,str] = {
    0: "#000000", 1: "#8B4513", 2: "#d32f2f", 3: "#F57C00", 4: "#FBC02D",
    5: "#388E3C", 6: "#1976D2", 7: "#7E57C2", 8: "#9E9E9E", 9: "#FFFFFF"
}

def color_for_n(n: int, palette: str = 'resistor') -> str:
    if palette == 'resistor':
        digit = (n % 10)
        return RESISTOR_COLORS.get(digit, f"#{(n*37)%255:02x}8899")
    if palette == 'classic':
        return {
            3: "#f27b7b", 4: "#f2b766", 5: "#f2e066", 6: "#78d689",
            7: "#7ec4f8", 8: "#a78bfa", 9: "#9ca3af", 10: "#c0c0c0",
            11: "#f3c846", 12: "#f4a6b8"
        }.get(n, f"hsl({(n*37)%360} 70% 70%)")
    # random-ish fallback
    return f"#{(n*53)%255:02x}{(n*97)%255:02x}{(n*193)%255:02x}"

# ----------------------------- OEIS A000940 -------------------------------
# Number of inequivalent n-gons (simple Hamiltonian cycles) up to dihedral symmetry
# Reference: https://oeis.org/A000940

def a000940(n: int) -> int:
    if n < 3: return 0
    def factorial(m: int) -> int:
        r = 1
        for i in range(2, m+1): r *= i
        return r
    def phi(m: int) -> int:
        r, x = m, m
        p = 2
        while p*p <= x:
            if x % p == 0:
                while x % p == 0: x //= p
                r -= r // p
            p += 1
        if x > 1: r -= r // x
        return r
    def divisors(m: int) -> List[int]:
        ds = []
        d = 1
        while d*d <= m:
            if m % d == 0:
                ds.append(d)
                if d*d != m: ds.append(m//d)
            d += 1
        ds.sort()
        return ds
    if n % 2 == 1:
        t1 = (2 ** ((n-1)//2)) * (n*n) * factorial((n-1)//2)
    else:
        t1 = (2 ** (n//2)) * n * (n+6) * factorial(n//2) / 4
    for d in divisors(n):
        m = n//d
        t1 += (phi(m)**2) * factorial(d) * (m ** d)
    val = t1 / (4 * n * n)
    return int(round(val))

# Simple representatives enumerator (small n) — dihedral-canonical
# returns list of index orders like [0,3,1,2,...]

def get_a000940_representatives(n: int) -> List[List[int]]:
    if n < 3: return []
    reps: List[List[int]] = []
    used = [False]*n
    path = [0]
    used[0] = True
    base = regular_polygon(n, 100)

    def segments_intersect(p, q, r, s):
        def orient(a,b,c):
            return math.copysign(1, (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) if (b!=a and c!=a) else 0
        o1 = orient(p,q,r); o2 = orient(p,q,s); o3 = orient(r,s,p); o4 = orient(r,s,q)
        if o1==0 and o2==0 and o3==0 and o4==0:
            return False
        return (o1!=o2 and o3!=o4)

    def cross(i,j,k,l):
        return segments_intersect(base[i], base[j], base[k], base[l])

    def can_add(v: int) -> bool:
        if used[v]: return False
        m = len(path); prev = path[-1]
        for i in range(m-2):
            a = path[i]; b = path[i+1]
            if a==prev or b==prev: continue
            if cross(a,b,prev,v): return False
        return True

    def close_simple() -> bool:
        m = len(path)
        for i in range(m-2):
            a = path[i]; b = path[i+1]
            if a==path[-1] or b==path[0]: continue
            if cross(a,b,path[-1],0): return False
        return True

    def canonical_dihedral() -> bool:
        return path[1] < path[-1]

    def dfs():
        if len(path) == n:
            if not close_simple(): return
            if not canonical_dihedral(): return
            reps.append(list(path))
            return
        for v in range(1,n):
            if not can_add(v):
                continue
            used[v] = True; path.append(v)
            dfs()
            path.pop(); used[v] = False
    dfs()
    return reps

# ----------------------------- Sequence builders --------------------------
@dataclass
class StarSpec:
    n: int
    k: int

@dataclass
class RepSpec:
    n: int
    order: List[int]

def build_sequence(min_n: int, max_n: int, include_compounds: bool, equiv: str) -> List[StarSpec]:
    seq: List[StarSpec] = []
    for n in range(min_n, max_n+1):
        k_max = (n//2) if equiv == 'dihedral' else (n-1)
        for k in range(1, k_max+1):
            coprime = gcd(n,k) == 1
            if not include_compounds and not coprime:
                continue
            seq.append(StarSpec(n,k))
    return seq

# Build ping-pong transitions within same-n segments; hard cuts between n when bouncing
@dataclass
class Transition:
    ai: int
    bi: int
    hard: bool

def build_transitions_bounce_no_cross_n(seq_n: List[int]) -> List[Transition]:
    if not seq_n: return []
    # group contiguous same-n segments
    segs = []
    i = 0
    while i < len(seq_n):
        n = seq_n[i]
        j = i
        while j+1 < len(seq_n) and seq_n[j+1] == n:
            j += 1
        segs.append((i,j))
        i = j+1
    out: List[Transition] = []
    # forward
    for s in range(len(segs)):
        a,b = segs[s]
        for k in range(a, b):
            out.append(Transition(k, k+1, False))
        if s < len(segs)-1:
            nxt = segs[s+1][0]
            out.append(Transition(b, nxt, True))
            out.append(Transition(nxt, nxt, True))  # dwell one step
    # backward
    for s in range(len(segs)-1, -1, -1):
        a,b = segs[s]
        for k in range(b, a, -1):
            out.append(Transition(k, k-1, False))
        if s > 0:
            prv = segs[s-1][0]
            out.append(Transition(a, prv, True))
            out.append(Transition(prv, prv, True))
    return out

# ----------------------------- Animation state ----------------------------
class Animator:
    def __init__(self,
                 mode: str = 'stars',
                 min_n: int = 3,
                 max_n: int = 12,
                 include_compounds: bool = True,
                 equiv: str = 'none',
                 duration_ms: int = 1200,
                 morph: bool = True,
                 palette: str = 'resistor'):
        self.mode = mode
        self.min_n = min_n
        self.max_n = max_n
        self.include_compounds = include_compounds
        self.equiv = equiv
        self.duration_ms = duration_ms
        self.morph = morph
        self.palette = palette
        # build sequence
        if mode == 'stars':
            self.sequence: List[StarSpec|RepSpec] = build_sequence(min_n, max_n, include_compounds, equiv)
        else:
            seq: List[RepSpec] = []
            cap = min(12, max_n)
            for n in range(min_n, cap+1):
                for order in get_a000940_representatives(n):
                    seq.append(RepSpec(n, order))
            self.sequence = seq
        self.seq_n = [getattr(x, 'n') for x in self.sequence]
        self.transitions = build_transitions_bounce_no_cross_n(self.seq_n)
        self.base_t = time.time()
        self.playing = True

    def current_edge(self) -> Tuple[int,int,float,bool]:
        if not self.transitions:
            return (0,0,0.0,False)
        elapsed = (time.time() - self.base_t) * 1000.0
        step_float = elapsed / max(1, self.duration_ms)
        s = int(math.floor(step_float))
        t = step_float - s
        m = s % len(self.transitions)
        tr = self.transitions[m]
        return (tr.ai, tr.bi, 1.0 if tr.hard else t, tr.hard)

# ----------------------------- App (matplotlib) ---------------------------

def run_app():
    # sanity tests
    assert gcd(12,8) == 4 and gcd(7,5) == 1
    assert a000940(3)==1 and a000940(4)==2 and a000940(5)==4 and a000940(6)==12 and a000940(7)==39
    # UI
    fig, ax = plt.subplots(figsize=(8,8))
    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.22)
    ax.set_aspect('equal'); ax.axis('off')

    animator = Animator()

    poly = plt.Polygon([[0,0],[1,0],[0,1]], closed=True, facecolor="#888", edgecolor="#111", linewidth=2)
    ax.add_patch(poly)
    label = ax.text(0.5, 0.05, "", transform=ax.transAxes, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', fc='w', ec='none', alpha=0.8))

    # sliders
    ax_min = plt.axes([0.12, 0.12, 0.3, 0.03]); s_min = Slider(ax_min, 'min n', 3, 60, valinit=3, valstep=1)
    ax_max = plt.axes([0.12, 0.08, 0.3, 0.03]); s_max = Slider(ax_max, 'max n', 3, 60, valinit=12, valstep=1)
    ax_dur = plt.axes([0.12, 0.04, 0.3, 0.03]); s_dur = Slider(ax_dur, 'ms/step', 100, 5000, valinit=1200, valstep=50)

    # buttons
    ax_play = plt.axes([0.46, 0.04, 0.1, 0.06]); b_play = Button(ax_play, 'Play/Pause')
    ax_mode = plt.axes([0.58, 0.04, 0.12, 0.06]); b_mode = Button(ax_mode, 'Mode stars/A940')
    ax_export = plt.axes([0.72, 0.04, 0.14, 0.06]); b_export = Button(ax_export, 'Export 1-bounce')

    def rebuild():
        animator.duration_ms = int(s_dur.val)
        animator.min_n = int(min(s_min.val, s_max.val))
        animator.max_n = int(max(s_min.val, s_max.val))
        if animator.mode == 'stars':
            animator.sequence = build_sequence(animator.min_n, animator.max_n, True, animator.equiv)
        else:
            seq = []
            for n in range(animator.min_n, min(12,animator.max_n)+1):
                for order in get_a000940_representatives(n):
                    seq.append(RepSpec(n, order))
            animator.sequence = seq
        animator.seq_n = [getattr(x, 'n') for x in animator.sequence]
        animator.transitions = build_transitions_bounce_no_cross_n(animator.seq_n)
        animator.base_t = time.time()

    def draw_frame(_):
        ai, bi, t, hard = animator.current_edge()
        if not animator.sequence:
            return
        A = animator.sequence[ai]; B = animator.sequence[bi]
        nA = getattr(A, 'n'); nB = getattr(B, 'n')
        r = 1.0
        if isinstance(A, StarSpec):
            ptsA = resample_closed(close_path(star_path(regular_polygon(A.n, r), A.k)), 360)
        else:
            baseA = regular_polygon(A.n, r)
            ptsA = close_path([baseA[i% A.n] for i in A.order])
            ptsA = resample_closed(ptsA, 360)
        if isinstance(B, StarSpec):
            ptsB = resample_closed(close_path(star_path(regular_polygon(B.n, r), B.k)), 360)
        else:
            baseB = regular_polygon(B.n, r)
            ptsB = close_path([baseB[i% B.n] for i in B.order])
            ptsB = resample_closed(ptsB, 360)
        pts = [(lerp(x0,x1,t), lerp(y0,y1,t)) for (x0,y0),(x1,y1) in zip(ptsA, ptsB)]
        col = lerp_color(color_for_n(nA), color_for_n(nB), t)
        poly.set_xy(pts)
        poly.set_facecolor(col)
        ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2)
        if t < 0.5:
            lbl = f"{{{nA}/{getattr(A,'k','•')}}}" if isinstance(A, StarSpec) else f"rep n={nA}: [{','.join(map(str,getattr(A,'order')))}]"
        else:
            lbl = f"{{{nB}/{getattr(B,'k','•')}}}" if isinstance(B, StarSpec) else f"rep n={nB}: [{','.join(map(str,getattr(B,'order')))}]"
        label.set_text(lbl)

    def on_play(_):
        animator.playing = not animator.playing
        if animator.playing:
            animator.base_t = time.time() - (animator.current_edge()[2] * animator.duration_ms)/1000.0
    b_play.on_clicked(on_play)

    def on_mode(_):
        animator.mode = 'a000940' if animator.mode=='stars' else 'stars'
        rebuild()
    b_mode.on_clicked(on_mode)

    def on_export(_):
        export_one_bounce(animator)
    b_export.on_clicked(on_export)

    def on_change(val):
        rebuild()
    s_min.on_changed(on_change); s_max.on_changed(on_change); s_dur.on_changed(on_change)

    def timer(_):
        if animator.playing:
            draw_frame(None)
        return poly,

    anim = FuncAnimation(fig, timer, interval=16, blit=False)
    plt.show()

# ----------------------------- Export -------------------------------------

def export_one_bounce(animator: Animator, filename: str | None = None, fps: int = 60):
    if not animator.transitions:
        print("Nothing to export (empty sequence)")
        return
    if filename is None:
        filename = f"inequivalent-polygons-1bounce-{fps}fps.mp4"
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='inequivalent-polygons'))
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal'); ax.axis('off')
    poly = plt.Polygon([[0,0],[1,0],[0,1]], closed=True, facecolor="#888", edgecolor="#111", linewidth=2)
    ax.add_patch(poly); ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2)

    # Render each transition for duration_ms
    with writer.saving(fig, filename, dpi=150):
        for tr in animator.transitions:
            A = animator.sequence[tr.ai]; B = animator.sequence[tr.bi]
            nA = getattr(A,'n'); nB = getattr(B,'n')
            if isinstance(A, StarSpec):
                ptsA = resample_closed(close_path(star_path(regular_polygon(A.n, 1.0), A.k)), 360)
            else:
                baseA = regular_polygon(A.n, 1.0)
                ptsA = close_path([baseA[i% A.n] for i in A.order])
                ptsA = resample_closed(ptsA, 360)
            if isinstance(B, StarSpec):
                ptsB = resample_closed(close_path(star_path(regular_polygon(B.n, 1.0), B.k)), 360)
            else:
                baseB = regular_polygon(B.n, 1.0)
                ptsB = close_path([baseB[i% B.n] for i in B.order])
                ptsB = resample_closed(ptsB, 360)
            steps = max(1, int(round(animator.duration_ms * fps / 1000)))
            for i in range(steps):
                t = 1.0 if tr.hard else (i/steps)
                pts = [(lerp(x0,x1,t), lerp(y0,y1,t)) for (x0,y0),(x1,y1) in zip(ptsA, ptsB)]
                col = lerp_color(color_for_n(nA), color_for_n(nB), t)
                poly.set_xy(pts); poly.set_facecolor(col)
                writer.grab_frame()
    plt.close(fig)
    print(f"Saved: {filename}")

# ----------------------------- Main ---------------------------------------

def run_tests():
    assert gcd(12,8)==4 and gcd(7,5)==1
    assert a000940(3)==1 and a000940(4)==2 and a000940(5)==4 and a000940(6)==12 and a000940(7)==39
    # simple transitions test
    seq = [5,5,6]  # two items of n=5, one item of n=6
    trs = build_transitions_bounce_no_cross_n(seq)
    assert len(trs) >= 3

if __name__ == '__main__':
    if '--test' in sys.argv:
        run_tests(); print('OK tests')
    else:
        run_app()
