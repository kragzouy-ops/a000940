#!/usr/bin/env python3
import os, sys, shutil
import gi


gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
gi.require_version("Vte", "2.91")


from gi.repository import Adw, Gtk, Vte, GLib


class AppTUI(Adw.Application):
def __init__(self):
super().__init__(application_id="com.example.GlancesGnome.TUI")
self.term: Vte.Terminal | None = None


def do_activate(self):
Adw.init()
win = Adw.ApplicationWindow(application=self, title="Glances (TUI)")
win.set_default_size(1000, 700)


hb = Adw.HeaderBar()
box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
box.append(hb)


self.term = Vte.Terminal()
box.append(self.term)
win.set_content(box)
win.present()


glances = shutil.which("glances") or "glances"
self.term.spawn_async(
Vte.PtyFlags.DEFAULT,
os.environ.get("HOME", "/"),
["/usr/bin/env", "bash", "-lc", glances],
[],
GLib.SpawnFlags.SEARCH_PATH,
None,
None,
-1,
None,
None,
)




def main(argv=None):
app = AppTUI()
return app.run(argv or sys.argv)


if __name__ == "__main__":
raise SystemExit(main())
