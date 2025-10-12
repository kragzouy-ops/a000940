import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paramètres de la lemniscate de Bernoulli
# (x^2 + y^2)^2 = 2a^2(x^2 - y^2)
a = 1.0

# Paramétrisation de la lemniscate
# x = a * cos(t) / (1 + sin^2(t))
# y = a * sin(t) * cos(t) / (1 + sin^2(t))
theta = np.linspace(0, 2 * np.pi, 1000)
x = a * np.cos(theta) / (1 + np.sin(theta)**2)
y = a * np.sin(theta) * np.cos(theta) / (1 + np.sin(theta)**2)

# Dynamique hamiltonienne simplifiée :
# On fait évoluer un point sur la courbe avec une "énergie" constante
# Ici, on suppose une vitesse angulaire constante (comme un oscillateur)

# Animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x, y, 'b-', label='Lemniscate de Bernoulli')
point, = ax.plot([], [], 'ro', label='Point hamiltonien')
ax.set_xlim(-1.2*a, 1.2*a)
ax.set_ylim(-1.2*a, 1.2*a)
ax.set_aspect('equal')
ax.legend()

# Initialisation de l'animation
def init():
    point.set_data([], [])
    return point,

# Fonction d'animation
def animate(i):
    idx = i % len(theta)
    point.set_data(x[idx], y[idx])
    return point,

ani = FuncAnimation(fig, animate, frames=len(theta), init_func=init,
                    interval=20, blit=True)

plt.title("Lemniscate de Bernoulli avec dynamique hamiltonienne")
plt.show()
