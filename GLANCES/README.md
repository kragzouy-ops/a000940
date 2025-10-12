# Glances GNOME


Encapsule Glances dans une application GNOME/Adwaita (WebUI) avec une variante TUI.


## Dépendances (dev système)
- Linux (GNOME ≥ 44 recommandé)
- PyGObject, GTK4, WebKitGTK, libadwaita (paquets distro, par ex. Debian/Ubuntu : `python3-gi gir1.2-gtk-4.0 gir1.2-webkit2-4.1 gir1.2-adw-1`)
- Glances (`pipx install glances`) si vous lancez hors Flatpak


## Démarrage rapide (dev)
```bash
make run
