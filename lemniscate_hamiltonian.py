import itertools
def unique_hamiltonian_cycles(n):
    # Génère tous les cycles hamiltoniens uniques (A000940) pour n sommets
    # On fixe le premier sommet à 0, on génère toutes les permutations des n-1 autres
    # et on élimine les cycles équivalents par inversion
    if n < 3:
        return []
    nodes = list(range(n))
    perms = set()
    for p in itertools.permutations(nodes[1:]):
        cycle = (0,) + p
        # Pour éviter les cycles inversés, on ne garde que si le cycle est "plus petit" que son inverse
        if cycle <= cycle[::-1]:
            perms.add(cycle)
    return list(perms)
def gray_code_sequence(n):
    # Génère la séquence Gray code de 0 à 2^n - 1
    return [i ^ (i >> 1) for i in range(2**n)]

def gray_code_path_indices(n):
    # Pour N points, on prend les indices du Gray code modulo N
    # Si N n'est pas une puissance de 2, on prend la plus proche puissance supérieure
    k = int(np.ceil(np.log2(n)))
    seq = gray_code_sequence(k)
    # Ramener à N points (en prenant le modulo)
    return [s % n for s in seq]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import math

# Paramètres de la lemniscate de Bernoulli
# (x^2 + y^2)^2 = 2a^2(x^2 - y^2)
a = 1.0

# Paramétrisation de la lemniscate
# x = a * cos(t) / (1 + sin^2(t))
# y = a * sin(t) * cos(t) / (1 + sin^2(t))
theta = np.linspace(0, 2 * np.pi, 1000)
x = a * np.cos(theta) / (1 + np.sin(theta)**2)
y = a * np.sin(theta) * np.cos(theta) / (1 + np.sin(theta)**2)

"""
Dynamique hamiltonienne améliorée :
On fait évoluer le point sur la lemniscate avec une vitesse variable,
comme si l'énergie totale (cinétique + potentielle) était conservée.
La vitesse dépend de la courbure de la courbe (plus rapide dans les zones plates, plus lente dans les zones serrées).
"""

# Calcul de la distance curviligne le long de la courbe
dx = np.gradient(x, theta)
dy = np.gradient(y, theta)
ds = np.sqrt(dx**2 + dy**2)
s = np.cumsum(ds)
s = s - s[0]  # normaliser à partir de 0
longueur_totale = s[-1]

# Définir une "énergie" totale et calculer la vitesse locale
# Pour simplifier, on prend E = 1/2 m v^2 + V(s), avec V(s) = 0
# Donc v(s) = constante, mais on peut moduler pour l'effet visuel
vitesse = 1.0 / (1 + 2 * np.abs(np.sin(2*theta)))  # vitesse variable selon la courbure
dt = ds / vitesse
t_cumul = np.cumsum(dt)
t_cumul = t_cumul - t_cumul[0]
t_cumul = t_cumul / t_cumul[-1]  # normaliser entre 0 et 1

# Animation


# Préparation des deux sous-graphes : lemniscate (gauche), cercle (droite)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')

# Lemniscate
ax.set_facecolor('black')
ax.plot(x, y, color='#00BFFF', lw=2)
point, = ax.plot([], [], 'o', color='#FFDD00', markersize=8)
ax.set_xlim(-1.2*a, 1.2*a)
ax.set_ylim(-1.2*a, 1.2*a)
ax.set_aspect('equal')
ax.set_title('')
ax.text(0.5, -0.13, 'Lemniscate', transform=ax.transAxes, fontsize=14, color='white', ha='center', va='top')
ax.tick_params(colors='gray')
for spine in ax.spines.values():
    spine.set_color('gray')

# Cercle
ax2.set_facecolor('black')
theta_c = np.linspace(0, 2 * np.pi, 1000)
x_c = a * np.cos(theta_c)
y_c = a * np.sin(theta_c)
ax2.plot(x_c, y_c, color='#00BFFF', lw=2)
point2, = ax2.plot([], [], 'o', color='#FFDD00', markersize=8)
ax2.set_xlim(-1.2*a, 1.2*a)
ax2.set_ylim(-1.2*a, 1.2*a)
ax2.set_aspect('equal')
ax2.set_title('')
ax2.text(0.5, -0.13, 'Cercle', transform=ax2.transAxes, fontsize=14, color='white', ha='center', va='top')
ax2.tick_params(colors='gray')
for spine in ax2.spines.values():
    spine.set_color('gray')


# Initialisation des valeurs de N (doit être avant tout usage)
N_init = 6
N_min = 3
N_max = 20

# Légende dynamique
def hamiltonian_cycle_count(n):
    # Nombre de cycles hamiltoniens uniques dans un graphe complet non orienté : (n-1)!/2
    return math.factorial(n-1)//2 if n > 2 else 1
def total_path_count(n):
    # Nombre total de chemins (permutations)
    return math.factorial(n)

legend_text = ax.text(0.02, 1.02, '', transform=ax.transAxes, fontsize=12, va='bottom', ha='left', color='#FFDD00', bbox=dict(facecolor='black', alpha=0.7, edgecolor='#FFDD00'))
legend_text2 = ax2.text(0.02, 1.02, '', transform=ax2.transAxes, fontsize=12, va='bottom', ha='left', color='#FFDD00', bbox=dict(facecolor='black', alpha=0.7, edgecolor='#FFDD00'))

def update_legend(N):
    txt = f"N = {N}\nPolygones hamiltoniens uniques : {hamiltonian_cycle_count(N):,}\nChemins possibles : {total_path_count(N):,}"
    legend_text.set_text(txt)
    legend_text2.set_text(txt)

update_legend(N_init)

def init():
    point.set_data([], [])
    return point,

def animate(i):
    # Trouver l'indice correspondant à l'instant t
    t = i / (len(theta)-1)
    idx = np.searchsorted(t_cumul, t)
    point.set_data(x[idx], y[idx])
    return point,



# --- Slider pour N ---
from matplotlib.widgets import Slider


# Initialisation des valeurs de N
N_init = 6
N_min = 3
N_max = 20

# Ajuster la position de la figure pour le slider

plt.subplots_adjust(bottom=0.28)
ax_slider = plt.axes([0.2, 0.13, 0.6, 0.04], facecolor='black')
slider_N = Slider(ax_slider, 'N sommets', N_min, N_max, valinit=N_init, valstep=1, color='#FFDD00')



# Slider pour permuter les cycles sur le cercle (créé dynamiquement si besoin)
ax_cycle = plt.axes([0.65, 0.10, 0.15, 0.04], facecolor='black')
slider_cycle = None

# Bouton Pause/Play
ax_pause = plt.axes([0.2, 0.05, 0.15, 0.05], facecolor='black')
btn_pause = Button(ax_pause, 'Pause', color='#222222', hovercolor='#444444')

# Bouton Export vidéo
ax_export = plt.axes([0.65, 0.05, 0.15, 0.05], facecolor='black')
btn_export = Button(ax_export, 'Exporter vidéo', color='#222222', hovercolor='#444444')


points_poly, = ax.plot([], [], 'o-', color='#FF8800', lw=2)
points_poly2, = ax2.plot([], [], 'o-', color='#FF8800', lw=2)

# Préparer les lignes pour toutes les arêtes (N*(N-1)/2 max)
from itertools import combinations
max_edges = (N_max * (N_max-1)) // 2
lines_edges = [ax.plot([], [], lw=1.5, color='#444444', alpha=0.3)[0] for _ in range(max_edges)]
lines_edges2 = [ax2.plot([], [], lw=1.5, color='#444444', alpha=0.3)[0] for _ in range(max_edges)]

N = N_init

def animate_poly(i):
    global N, slider_cycle
    N = int(slider_N.val)
    update_legend(N)
    t = i / (len(theta)-1)
    # --- Lemniscate ---
    idxs = [(np.searchsorted(t_cumul, (t + k/N) % 1.0)) for k in range(N)]
    # Palette des couleurs de résistance
    resistor_colors = [
        "#000000",  # Noir
        "#8B4513",  # Marron
        "#FF0000",  # Rouge
        "#FFA500",  # Orange
        "#FFFF00",  # Jaune
        "#008000",  # Vert
        "#0000FF",  # Bleu
        "#8B00FF",  # Violet
        "#808080",  # Gris
        "#FFFFFF",  # Blanc
    ]
    # Chemin dynamique sur la lemniscate
    x_poly = [x[j] for j in idxs] + [x[idxs[0]]]
    y_poly = [y[j] for j in idxs] + [y[idxs[0]]]
    # Efface les anciens segments dynamiques (traces résiduelles)
    if hasattr(animate_poly, 'dynamic_lines'):
        for l in animate_poly.dynamic_lines:
            l.remove()
    animate_poly.dynamic_lines = []
    # Trace chaque segment avec la couleur du code de résistance
    for k in range(N):
        color = resistor_colors[k % len(resistor_colors)]
        l, = ax.plot([x_poly[k], x_poly[k+1]], [y_poly[k], y_poly[k+1]], '-', color=color, lw=2, zorder=5)
        animate_poly.dynamic_lines.append(l)
    idx = np.searchsorted(t_cumul, t)
    point.set_data(x[idx], y[idx])
    # --- Cercle animé ---
    # Décalage angulaire global pour l'animation
    angle_offset = 2 * np.pi * t
    idxs2 = [int(((k/N) + t) % 1.0 * len(theta_c)) for k in range(N)]
    # Chemin dynamique sur le cercle
    x_poly2 = [x_c[j] for j in idxs2] + [x_c[idxs2[0]]]
    y_poly2 = [y_c[j] for j in idxs2] + [y_c[idxs2[0]]]
    if hasattr(animate_poly, 'dynamic_lines2'):
        for l in animate_poly.dynamic_lines2:
            l.remove()
    animate_poly.dynamic_lines2 = []
    for k in range(N):
        color = resistor_colors[k % len(resistor_colors)]
        l2, = ax2.plot([x_poly2[k], x_poly2[k+1]], [y_poly2[k], y_poly2[k+1]], '-', color=color, lw=2, zorder=5)
        animate_poly.dynamic_lines2.append(l2)
    idx2 = int((t % 1.0) * len(theta_c))
    point2.set_data(x_c[idx2], y_c[idx2])

    # Couleurs différentes pour chaque arête
    color_map = plt.cm.get_cmap('hsv', N*(N-1)//2+1)
    combs = list(combinations(range(N), 2))
    for k, line in enumerate(lines_edges):
        if k < len(combs):
            i1, i2 = combs[k]
            line.set_data([x[idxs[i1]], x[idxs[i2]]], [y[idxs[i1]], y[idxs[i2]]])
            line.set_color(color_map(k))
            line.set_alpha(0.2)
            line.set_zorder(1)
            line.set_visible(True)
        else:
            line.set_visible(False)
    for k, line in enumerate(lines_edges2):
        if k < len(combs):
            i1, i2 = combs[k]
            line.set_data([x_c[idxs2[i1]], x_c[idxs2[i2]]], [y_c[idxs2[i1]], y_c[idxs2[i2]]])
            line.set_color(color_map(k))
            line.set_alpha(0.2)
            line.set_zorder(1)
            line.set_visible(True)
        else:
            line.set_visible(False)

    # Affichage de tous les cycles hamiltoniens uniques (A000940) en bleu clair sur la lemniscate
    if not hasattr(animate_poly, 'hamiltonian_lines') or len(animate_poly.hamiltonian_lines) != hamiltonian_cycle_count(N):
        # Supprimer les anciens si besoin
        if hasattr(animate_poly, 'hamiltonian_lines'):
            for l in animate_poly.hamiltonian_lines:
                l.remove()
            for l in animate_poly.hamiltonian_lines2:
                l.remove()
        # Créer les nouveaux
        cycles = unique_hamiltonian_cycles(N)
        animate_poly.hamiltonian_lines = [ax.plot([], [], color='#00FFCC', lw=1.5, alpha=0.7, zorder=2)[0] for _ in cycles]
        animate_poly.hamiltonian_lines2 = [ax2.plot([], [], color='#00FFCC', lw=1.5, alpha=0.7, zorder=2)[0] for _ in cycles]
        animate_poly.hamiltonian_cycles = cycles
        # Créer ou mettre à jour le slider de cycle
        if len(cycles) > 0:
            if slider_cycle is None:
                slider_cycle = Slider(ax_cycle, 'Cycle', 0, len(cycles)-1, valinit=0, valstep=1)
            else:
                slider_cycle.valmin = 0
                slider_cycle.valmax = len(cycles)-1
                slider_cycle.valstep = 1 if len(cycles) > 1 else None
                slider_cycle.set_val(0)
            slider_cycle.ax.set_visible(True)
        else:
            if slider_cycle is not None:
                slider_cycle.ax.set_visible(False)
    # Mettre à jour les lignes
    # Animation automatique du slider du cercle
    if slider_cycle is not None and slider_cycle.ax.get_visible() and hasattr(animate_poly, 'hamiltonian_cycles'):
        n_cycles = len(animate_poly.hamiltonian_cycles)
        # Si l'utilisateur n'est pas en train de manipuler le slider, on l'anime
        if not hasattr(slider_cycle, '_dragging') or not slider_cycle._dragging:
            slider_cycle.set_val(i % n_cycles)

    # Palette des couleurs de résistance
    resistor_colors = [
        "#000000",  # Noir
        "#8B4513",  # Marron
        "#FF0000",  # Rouge
        "#FFA500",  # Orange
        "#FFFF00",  # Jaune
        "#008000",  # Vert
        "#0000FF",  # Bleu
        "#8B00FF",  # Violet
        "#808080",  # Gris
        "#FFFFFF",  # Blanc
    ]
    for idx_cyc, (l, l2, cyc) in enumerate(zip(animate_poly.hamiltonian_lines, animate_poly.hamiltonian_lines2, animate_poly.hamiltonian_cycles)):
        color = resistor_colors[idx_cyc % len(resistor_colors)]
        # Lemniscate : tous visibles
        x_cy = [x[idxs[j]] for j in cyc] + [x[idxs[cyc[0]]]]
        y_cy = [y[idxs[j]] for j in cyc] + [y[idxs[cyc[0]]]]
        l.set_data(x_cy, y_cy)
        l.set_color(color)
        l.set_visible(True)
        # Cercle : un seul visible (selon slider)
        if slider_cycle is not None and idx_cyc == int(slider_cycle.val):
            x_cy2 = [x_c[idxs2[j]] for j in cyc] + [x_c[idxs2[cyc[0]]]]
            y_cy2 = [y_c[idxs2[j]] for j in cyc] + [y_c[idxs2[cyc[0]]]]
            l2.set_data(x_cy2, y_cy2)
            l2.set_color(color)
            l2.set_visible(True)
        else:
            l2.set_visible(False)

    # Affichage fluorescent de la trace selon la suite A000940 (Gray code)
    gray_path = gray_code_path_indices(N)
    x_gray = [x[idxs[i]] for i in gray_path]
    y_gray = [y[idxs[i]] for i in gray_path]
    if not hasattr(animate_poly, 'gray_line'):
        animate_poly.gray_line, = ax.plot([], [], color='#39FF14', lw=3, alpha=0.95, zorder=3)
        animate_poly.gray_line2, = ax2.plot([], [], color='#39FF14', lw=3, alpha=0.95, zorder=3)
    animate_poly.gray_line.set_data(x_gray, y_gray)
    animate_poly.gray_line.set_visible(True)
    # Cercle : trace Gray
    x_gray2 = [x_c[idxs2[i]] for i in gray_path]
    y_gray2 = [y_c[idxs2[i]] for i in gray_path]
    animate_poly.gray_line2.set_data(x_gray2, y_gray2)
    animate_poly.gray_line2.set_visible(True)


    # Affichage fluorescent de la trace selon la suite A000940 (Gray code)
    gray_path = gray_code_path_indices(N)
    x_gray = [x[idxs[i]] for i in gray_path]
    y_gray = [y[idxs[i]] for i in gray_path]
    if not hasattr(animate_poly, 'gray_line'):
        animate_poly.gray_line, = ax.plot([], [], color='#39FF14', lw=3, alpha=0.95, zorder=3)
        animate_poly.gray_line2, = ax2.plot([], [], color='#39FF14', lw=3, alpha=0.95, zorder=3)
    animate_poly.gray_line.set_data(x_gray, y_gray)
    animate_poly.gray_line.set_visible(True)
    # Cercle : trace Gray
    x_gray2 = [x_c[idxs2[i]] for i in gray_path]
    y_gray2 = [y_c[idxs2[i]] for i in gray_path]
    animate_poly.gray_line2.set_data(x_gray2, y_gray2)
    animate_poly.gray_line2.set_visible(True)

    # Mettre à jour le label du slider
    if slider_cycle is not None and slider_cycle.ax.get_visible() and hasattr(animate_poly, 'hamiltonian_cycles'):
        slider_cycle.label.set_text(f'Cycle ({int(slider_cycle.val)+1}/{max(1,len(animate_poly.hamiltonian_cycles))})')

    return (point, points_poly, animate_poly.gray_line) + tuple(animate_poly.hamiltonian_lines) + tuple(lines_edges) + \
           (point2, points_poly2, animate_poly.gray_line2) + tuple(animate_poly.hamiltonian_lines2) + tuple(lines_edges2)
# Permettre à l'utilisateur de reprendre le contrôle manuel du slider du cercle
def on_slider_cycle_press(event):
    global slider_cycle
    if slider_cycle is not None and event.inaxes == slider_cycle.ax:
        slider_cycle._dragging = True

def on_slider_cycle_release(event):
    global slider_cycle
    if slider_cycle is not None and event.inaxes == slider_cycle.ax:
        slider_cycle._dragging = False

# Initialiser le flag _dragging et connecter les événements après la création du slider
def setup_slider_cycle_events():
    global slider_cycle
    if slider_cycle is not None:
        slider_cycle._dragging = False
        slider_cycle.on_changed(lambda val: None)  # pour forcer la création de l'objet
        slider_cycle.ax.figure.canvas.mpl_connect('button_press_event', on_slider_cycle_press)
        slider_cycle.ax.figure.canvas.mpl_connect('button_release_event', on_slider_cycle_release)

setup_slider_cycle_events()


# --- Animation avec vitesse variable selon la distance minimale entre points ---
class DynamicSpeedAnimation:
    def __init__(self, fig, func, frames, init_func=None, blit=True, base_interval=20, min_interval=5, max_interval=100):
        self.fig = fig
        self.func = func
        self.frames = frames
        self.init_func = init_func
        self.blit = blit
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.frame_idx = 0
        self.ani = None
        self.running = True
        self._start()

    def _start(self):
        if self.init_func:
            self.init_func()
        self._timer = self.fig.canvas.new_timer(interval=self.base_interval)
        self._timer.add_callback(self._step)
        self._timer.start()

    def _step(self):
        if not self.running:
            return
        artists = self.func(self.frame_idx)
        if self.blit:
            self.fig.canvas.draw_idle()
        self.frame_idx = (self.frame_idx + 1) % self.frames
        # Calculer la distance minimale entre les points N
        N = int(slider_N.val)
        t = self.frame_idx / (len(theta)-1)
        idxs = [(np.searchsorted(t_cumul, (t + k/N) % 1.0)) for k in range(N)]
        pts = np.array([[x[j], y[j]] for j in idxs])
        dists = np.sqrt(np.sum((pts[None,:,:] - pts[:,None,:])**2, axis=-1))
        min_dist = np.min(dists[np.nonzero(dists)])
        # Ajuster l'intervalle : plus les points sont proches, plus on ralentit
        # (intervalle = max_interval quand min_dist est petit, min_interval quand min_dist est grand)
        norm = (min_dist - 0.0) / (2*a - 0.0)
        norm = np.clip(norm, 0, 1)
        interval = self.max_interval - norm * (self.max_interval - self.min_interval)
        self._timer.interval = int(interval)
        self._timer.start()

    def stop(self):
        self.running = False


ani = DynamicSpeedAnimation(fig, animate_poly, frames=len(theta), init_func=init,
                            base_interval=20, min_interval=5, max_interval=120)

# Gestion du bouton Pause/Play
def on_pause_clicked(event):
    if ani.running:
        ani.stop()
        btn_pause.label.set_text('Play')
    else:
        ani.running = True
        ani._timer.start()
        btn_pause.label.set_text('Pause')
btn_pause.on_clicked(on_pause_clicked)

# Gestion du bouton Export vidéo
def on_export_clicked(event):
    from matplotlib.animation import FFMpegWriter
    btn_export.label.set_text('Export...')
    # Créer une animation standard pour l'export (une seule boucle complète)
    export_anim = FuncAnimation(
        fig,
        animate_poly,
        frames=len(theta),
        init_func=init,
        blit=True,
        repeat=False  # Ne boucle pas
    )
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Hamiltonian Lemniscate'))
    try:
        export_anim.save('lemniscate_hamiltonian.mp4', writer=writer)
        btn_export.label.set_text('Vidéo OK')
    except Exception as e:
        btn_export.label.set_text('Erreur')
        print('Erreur export vidéo:', e)
btn_export.on_clicked(on_export_clicked)

plt.title("Lemniscate de Bernoulli avec polygone hamiltonien dynamique")
plt.show()
