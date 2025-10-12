import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math

st.set_page_config(layout="wide", page_title="Lemniscate Hamiltonienne")
st.title("Lemniscate Hamiltonienne et Polygones Hamiltoniens")

# Paramètres
N_min, N_max = 3, 10
N = st.slider("Nombre de sommets (N)", N_min, N_max, 6, 1)
frame = st.slider("Animation (frame)", 0, 999, 0, 1)
show_all_cycles = st.checkbox("Afficher tous les cycles hamiltoniens sur la lemniscate", value=True)
cycle_idx = st.slider("Cycle unique sur le cercle", 0, max(0, math.factorial(N-1)//2-1), 0, 1) if N > 3 else 0

# Lemniscate
st.write(":blue[À gauche : lemniscate de Bernoulli]  |  :orange[À droite : cercle]  |  :green[Trace Gray]  |  :violet[Cycles hamiltoniens]  |  :yellow[Polygone dynamique]")
a = 1.0
theta = np.linspace(0, 2 * np.pi, 1000)
x = a * np.cos(theta) / (1 + np.sin(theta)**2)
y = a * np.sin(theta) * np.cos(theta) / (1 + np.sin(theta)**2)

# Cercle
x_c = a * np.cos(theta)
y_c = a * np.sin(theta)

def unique_hamiltonian_cycles(n):
    if n < 3:
        return []
    nodes = list(range(n))
    perms = set()
    for p in itertools.permutations(nodes[1:]):
        cycle = (0,) + p
        if cycle <= cycle[::-1]:
            perms.add(cycle)
    return list(perms)

def gray_code_sequence(n):
    return [i ^ (i >> 1) for i in range(2**n)]

def gray_code_path_indices(n):
    k = int(np.ceil(np.log2(n)))
    seq = gray_code_sequence(k)
    return [s % n for s in seq]

def plot_fig(N, frame, show_all_cycles, cycle_idx):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')
    # Lemniscate
    ax.set_facecolor('black')
    ax.plot(x, y, color='#00BFFF', lw=2)
    # Cercle
    ax2.set_facecolor('black')
    ax2.plot(x_c, y_c, color='#00BFFF', lw=2)
    # Animation
    t = frame / 999
    idxs = [(np.searchsorted(theta, (t + k/N)*2*np.pi % (2*np.pi))) for k in range(N)]
    idxs2 = [(int(((k/N) + t) % 1.0 * len(theta))) for k in range(N)]
    # Polygone dynamique
    ax.plot([x[j] for j in idxs]+[x[idxs[0]]], [y[j] for j in idxs]+[y[idxs[0]]], 'o-', color='#FF8800', lw=2)
    ax2.plot([x_c[j] for j in idxs2]+[x_c[idxs2[0]]], [y_c[j] for j in idxs2]+[y_c[idxs2[0]]], 'o-', color='#FF8800', lw=2)
    # Points
    ax.plot(x[idxs], y[idxs], 'o', color='#FFDD00', markersize=8)
    ax2.plot(x_c[idxs2], y_c[idxs2], 'o', color='#FFDD00', markersize=8)
    # Cycles hamiltoniens
    cycles = unique_hamiltonian_cycles(N)
    if show_all_cycles:
        for cyc in cycles:
            ax.plot([x[idxs[j]] for j in cyc]+[x[idxs[cyc[0]]]], [y[idxs[j]] for j in cyc]+[y[idxs[cyc[0]]]], color='#00FFCC', lw=1.2, alpha=0.7)
    if cycles:
        cyc = cycles[cycle_idx % len(cycles)]
        ax2.plot([x_c[idxs2[j]] for j in cyc]+[x_c[idxs2[cyc[0]]]], [y_c[idxs2[j]] for j in cyc]+[y_c[idxs2[cyc[0]]]], color='#00FFCC', lw=2, alpha=0.9)
    # Trace Gray
    gray_path = gray_code_path_indices(N)
    ax.plot([x[idxs[i]] for i in gray_path], [y[idxs[i]] for i in gray_path], color='#39FF14', lw=2.5, alpha=0.95)
    ax2.plot([x_c[idxs2[i]] for i in gray_path], [y_c[idxs2[i]] for i in gray_path], color='#39FF14', lw=2.5, alpha=0.95)
    # Nettoyage
    for a in (ax, ax2):
        a.set_xlim(-1.2*a.get_xlim()[1], 1.2*a.get_xlim()[1])
        a.set_ylim(-1.2*a.get_ylim()[1], 1.2*a.get_ylim()[1])
        a.set_aspect('equal')
        a.axis('off')
    ax.text(0.5, -0.13, 'Lemniscate', transform=ax.transAxes, fontsize=14, color='white', ha='center', va='top')
    ax2.text(0.5, -0.13, 'Cercle', transform=ax2.transAxes, fontsize=14, color='white', ha='center', va='top')
    return fig

fig = plot_fig(N, frame, show_all_cycles, cycle_idx)
st.pyplot(fig)

st.markdown(f"**N = {N}**  ")
st.markdown(f"Polygones hamiltoniens uniques : **{math.factorial(N-1)//2 if N>2 else 1:,}**  ")
st.markdown(f"Chemins possibles : **{math.factorial(N):,}**  ")
