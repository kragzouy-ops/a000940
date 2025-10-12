#!/usr/bin/env python3
import os, sys, shutil, subprocess, socket, signal, time
import gi
gi.require_version("Gtk", "3.0")
try:
    gi.require_version("WebKit2", "4.1")
except ValueError:
    gi.require_version("WebKit2", "4.0")
from gi.repository import Gtk, GLib, Gio, WebKit2

GLANCES_PORT = 61208
GLANCES_URL  = f"http://127.0.0.1:{GLANCES_PORT}"

def wait_for_port(host, port, timeout_s=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False

class GlancesGnome(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.GlancesGnome",
                         flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.window = None
        self.proc = None
        self.glances_cmd = shutil.which("glances")

        # Arrêt propre quand l’app se ferme
        self.connect("shutdown", self.on_shutdown)

    def do_startup(self):
        Gtk.Application.do_startup(self)

        # Actions d’application
        action_quit = Gio.SimpleAction.new("quit", None)
        action_quit.connect("activate", lambda *_: self.quit())
        self.add_action(action_quit)
        self.set_accels_for_action("app.quit", ["<Primary>q"])

        action_open_ext = Gio.SimpleAction.new("openext", None)
        action_open_ext.connect("activate", self.on_open_external)
        self.add_action(action_open_ext)

        action_reload = Gio.SimpleAction.new("reload", None)
        action_reload.connect("activate", self.on_reload)
        self.add_action(action_reload)

    def do_activate(self):
        if not self.window:
            self.window = Gtk.ApplicationWindow(application=self)
            self.window.set_title("Glances")
            self.window.set_default_size(1100, 720)

            header = Gtk.HeaderBar()
            self.window.set_titlebar(header)

            btn_reload = Gtk.Button.new_from_icon_name("view-refresh-symbolic")
            btn_reload.connect("clicked", self.on_reload)
            header.pack_start(btn_reload)

            btn_openext = Gtk.Button.new_from_icon_name("window-new-symbolic")
            btn_openext.connect("clicked", self.on_open_external)
            header.pack_end(btn_openext)

            # Zone de contenu
            self.stack = Gtk.Stack()
            self.stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
            self.window.set_child(self.stack)

            # Page "loading"
            self.loading_lbl = Gtk.Label(label="Lancement de Glances…")
            self.loading_lbl.add_css_class("title-2")
            self.stack.add_named(self.loading_lbl, "loading")

            # Page Web
            self.webview = WebKit2.WebView()
            self.stack.add_named(self.webview, "web")

            # Page erreur
            self.error_lbl = Gtk.Label()
            self.error_lbl.add_css_class("error")
            self.error_lbl.set_wrap(True)
            self.stack.add_named(self.error_lbl, "error")

            self.window.present()
            GLib.idle_add(self.start_and_load)
        else:
            self.window.present()

    def start_and_load(self):
        if not self.glances_cmd:
            self.show_error(
                "Glances n’est pas installé dans cet environnement.\n"
                "Installe-le par exemple avec:\n"
                "  pipx install glances\nou emballe via Flatpak (manifest ci-dessous)."
            )
            return False

        # Démarre le serveur web de Glances
        try:
            # --disable-webui-auth pour éviter la demande de mot de passe en local
            self.proc = subprocess.Popen(
                [self.glances_cmd, "-w", "--quiet", "--disable-webui-auth"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None
            )
        except Exception as e:
            self.show_error(f"Impossible de lancer Glances: {e}")
            return False

        # Attends que le port écoute puis charge l’UI
        def poll_ready():
            if wait_for_port("127.0.0.1", GLANCES_PORT, timeout_s=0.2):
                self.webview.load_uri(GLANCES_URL)
                self.stack.set_visible_child_name("web")
                return False
            # Si le process s’est arrêté entre-temps -> erreur
            if self.proc and self.proc.poll() is not None:
                self.show_error("Le processus Glances s’est arrêté de façon inattendue.")
                return False
            return True

        self.stack.set_visible_child_name("loading")
        GLib.timeout_add(200, poll_ready)
        return False

    def on_open_external(self, *_):
        Gio.AppInfo.launch_default_for_uri(GLANCES_URL, None)

    def on_reload(self, *_):
        if self.stack.get_visible_child_name() == "web":
            self.webview.reload()
        else:
            # réessaye de lancer/charger
            self.start_and_load()

    def show_error(self, msg: str):
        self.error_lbl.set_text(msg)
        self.stack.set_visible_child_name("error")

    def on_shutdown(self, *_):
        # Termine proprement Glances
        if self.proc and self.proc.poll() is None:
            try:
                if hasattr(os, "getpgid") and hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                else:
                    self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if hasattr(os, "getpgid") and hasattr(os, "killpg"):
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    else:
                        self.proc.kill()
            except Exception:
                pass

def main():
    app = GlancesGnome()
    return app.run(sys.argv)

if __name__ == "__main__":
    raise SystemExit(main())

