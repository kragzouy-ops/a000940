# Installe le script et crée un lanceur simple
install -Dm755 ./main.py ~/.local/bin/glances-gnome
desktop-file-install --dir="$HOME/.local/share/applications" com.example.GlancesGnome.desktop
update-desktop-database "$HOME/.local/share/applications"
