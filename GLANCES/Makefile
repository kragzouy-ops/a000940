APP_ID = com.example.GlancesGnome
PREFIX ?= $(HOME)/.local


.PHONY: run install-user uninstall-user desktop-install desktop-uninstall flatpak flatpak-run


run:
    python -m glances_gnome


install-user:
    pip install --user -e .


uninstall-user:
    pip uninstall -y glances-gnome || true


desktop-install:
    install -Dm644 data/$(APP_ID).desktop $(PREFIX)/share/applications/$(APP_ID).desktop
    install -Dm644 data/icons/$(APP_ID).png $(PREFIX)/share/icons/hicolor/512x512/apps/$(APP_ID).png
    update-desktop-database $(PREFIX)/share/applications || true


desktop-uninstall:
    rm -f $(PREFIX)/share/applications/$(APP_ID).desktop
    rm -f $(PREFIX)/share/icons/hicolor/512x512/apps/$(APP_ID).png
    update-desktop-database $(PREFIX)/share/applications || true


flatpak:
    flatpak-builder --user --install --force-clean build-dir flatpak/$(APP_ID).yml


flatpak-run:
    flatpak run $(APP_ID)
