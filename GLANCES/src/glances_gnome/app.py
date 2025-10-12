#!/usr/bin/env python3
import os, sys, shutil, subprocess, socket, signal, time
import gi


gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
# WebKit 4.1 si dispo (Fedora/GNOME 46), sinon 4.0
try:
gi.require_version("WebKit2", "4.1")
except ValueError:
gi.require_version("WebKit2", "4.0")


from gi.repository import Adw, Gtk, GLib, Gio, WebKit2


GLANCES_PORT = int(os.getenv("GLANCES_PORT", 61208))
GLANCES_URL = f"http://127.0.0.1:{GLANCES_PORT}"




def wait_for_port(host: str, port: int, timeout_s: float = 10.0) -> bool:
t0 = time.time()
while time.time() - t0 < timeout_s:
try:
with socket.create_connection((host, port), timeout=0.2):
return True
except OSError:
time.sleep(0.1)
return False




class App(Adw.Application):
def __init__(self):
super().__init__(
application_id="com.example.GlancesGnome",
flags=Gio.ApplicationFlags.FLAGS_NONE,
)
self.window: Adw.ApplicationWindow | None = None
self.proc: subprocess.Popen | None = None
self.glances_cmd = shutil.which("glances")
self.webview: WebKit2.WebView | None = None
self.stack: Gtk.Stack | None = None
self.toast_overlay: Adw.ToastOverlay | None = None
self.connect("shutdown", self.on_shutdown)


# Actions / raccourcis
def do_startup(self):
Adw.Application.do_startup(self)


def add_action(name, cb, accel=None):
act = Gio.SimpleAction.new(name, None)
act.connect("activate", cb)
self.add_action(act)
if accel:
self.set_accels_for_action(f"app.{name}", [accel])


add_action("quit", lambda *_: self.quit(), "<Primary>q")
add_action("reload", self.on_reload, "<Primary>r")
raise SystemExit(main())
