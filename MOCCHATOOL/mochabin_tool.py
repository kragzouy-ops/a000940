#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import typer
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from rich.prompt import Prompt
from rich.layout import Layout
from rich.columns import Columns
import sys
import subprocess
from io import StringIO
from rich.table import Table
from rich.syntax import Syntax
import os
import shlex
try:
    import serial
    import serial.tools.list_ports
    import serial.tools.list_ports_common
except ImportError:
    serial = None

DEFAULT_PROMPT = r"=> "
DEFAULT_TIMEOUT = 30

import time
from typing import Optional, Iterable
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn
try:
    import fdpexpect
except ImportError:
    fdpexpect = None

# Define or import these if not already present
DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200

app = typer.Typer()
console = Console()

# --- Preinit: venv check, requirements install, and self-restart ---
def _preinit():
    import subprocess
    import shutil
    import pathlib
    import sys
    import os
    VENV_PATH = os.path.join(os.path.dirname(__file__), '.venv')
    REQUIREMENTS = ['typer[all]', 'rich', 'pyserial', 'pexpect']
    REDUNDANCY_ENV = 'MOCHABIN_TOOL_BOOTSTRAPPED'
    # Only run if not already bootstrapped
    if os.environ.get(REDUNDANCY_ENV) == '1':
        return
    # Check if running inside venv
    in_venv = sys.prefix != sys.base_prefix or hasattr(sys, 'real_prefix')
    if not in_venv:
        # Create venv if missing
        if not os.path.isdir(VENV_PATH):
            subprocess.check_call([sys.executable, '-m', 'venv', VENV_PATH])
        # Re-exec inside venv
        pybin = os.path.join(VENV_PATH, 'bin', 'python')
        os.environ[REDUNDANCY_ENV] = '1'
        os.execv(pybin, [pybin] + sys.argv)
    # In venv: check requirements
    try:
        import typer, rich, serial, pexpect
    except ImportError:
        # Install requirements
        pip = sys.executable
        subprocess.check_call([pip, '-m', 'pip', 'install'] + REQUIREMENTS)
        # Re-exec to reload modules
        os.environ[REDUNDANCY_ENV] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)

# Only run preinit if not imported as a module
if __name__ == "__main__":
    _preinit()


# --- Preinit: venv check, requirements install, and self-restart ---
def _preinit():
    import subprocess
    import shutil
    import pathlib
    import sys
    import os
    VENV_PATH = os.path.join(os.path.dirname(__file__), '.venv')
    REQUIREMENTS = ['typer[all]', 'rich', 'pyserial', 'pexpect']
    REDUNDANCY_ENV = 'MOCHABIN_TOOL_BOOTSTRAPPED'
    # Only run if not already bootstrapped
    if os.environ.get(REDUNDANCY_ENV) == '1':
        return
    # Check if running inside venv
    in_venv = sys.prefix != sys.base_prefix or hasattr(sys, 'real_prefix')
    if not in_venv:
        # Create venv if missing
        if not os.path.isdir(VENV_PATH):
            subprocess.check_call([sys.executable, '-m', 'venv', VENV_PATH])
        # Re-exec inside venv
        pybin = os.path.join(VENV_PATH, 'bin', 'python')
        os.environ[REDUNDANCY_ENV] = '1'
        os.execv(pybin, [pybin] + sys.argv)
    # In venv: check requirements
    try:
        import typer, rich, serial, pexpect
    except ImportError:
        # Install requirements
        pip = sys.executable
        subprocess.check_call([pip, '-m', 'pip', 'install'] + REQUIREMENTS)
        # Re-exec to reload modules
        os.environ[REDUNDANCY_ENV] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)

# Only run preinit if not imported as a module
if __name__ == "__main__":
    _preinit()

@app.command("console-ui")
def cmd_console_ui():
    """Interactive console UI: select commands and view output in a split-pane display."""

    commands = [
        ("list-ports", "List available serial devices."),
        ("console", "Interactive serial console with logging, exit keys, and prefix-escape."),
        ("log", "Non-interactive capture of serial output to a file."),
        ("break", "Spam CR to stop autoboot and land at U-Boot prompt."),
        ("run", "Run a U-Boot script or built-in recipe (flash/netboot/env/reset)."),
        ("kwboot", "UART boot a temporary U-Boot image."),
        ("doctor", "Environment checks (permissions, kwboot in PATH, ports)."),
        ("help", "Show this cheatsheet of commands and examples."),
    ]
    output_height = 10
    console_height = 5
    layout = Layout()
    layout.split_column(
        Layout(name="menu", size=12),
        Layout(name="output", size=output_height),
        Layout(name="console", size=console_height),
        Layout(name="status", size=1)
    )
    output_buffers = [["[dim]Command output will appear here.[/dim]"]]
    output_scrolls = [0]
    output_idx = 0
    console_text = Text.from_markup("[dim]Interactive console will appear here when supported.[/dim]")
    status_text = Text.from_markup("")
    menu_conn_info = {"info": None}
    import sys, termios, tty, select
    selected = 0
    with Live(layout, refresh_per_second=10, screen=True) as live:
        while True:
            layout["menu"].update(render_menu(selected))
            output_lines = []
            if output_buffers:
                for line in output_buffers[output_idx]:
                    output_lines.extend(line.splitlines() or [""])
            visible_lines = layout["output"].size - 2
            max_scroll = max(0, len(output_lines) - visible_lines)
            output_scrolls[output_idx] = min(output_scrolls[output_idx], max_scroll)
            output_scrolls[output_idx] = max(0, output_scrolls[output_idx])
            shown = output_lines[output_scrolls[output_idx]:output_scrolls[output_idx]+visible_lines]
            slider_height = visible_lines
            slider_pos = int((output_scrolls[output_idx] / max(1, max_scroll)) * (slider_height-1)) if max_scroll else 0
            slider = ["│" for _ in range(slider_height)]
            if slider:
                slider[slider_pos] = "█"
            slider_text = "\n".join(slider)
            output_panel = Panel(
                Columns([
                    Text.from_ansi("\n".join(shown)),
                    Text(slider_text, style="bright_black")
                ], equal=True, expand=True),
                title=f"Output [{output_idx+1}/{len(output_buffers)}] ([/]: prev/next)",
                border_style="magenta"
            )
            layout["output"].update(output_panel)
            layout["console"].update(Panel(console_text, title="Console", border_style="blue"))
            layout["status"].update(status_text)
            live.update(layout)
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                rlist, _, _ = select.select([fd], [], [], 0.2)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == "\x1b":
                        # Possible escape sequence (arrow keys, etc.)
                        seq = sys.stdin.read(2)
                        if seq == "[A":
                            selected = (selected - 1) % len(commands)
                        elif seq == "[B":
                            selected = (selected + 1) % len(commands)
                        # Optionally handle left/right or other keys here
                    elif ch in ('\r', '\n'):
                        # Enter key (handle both CR and LF)
                        cmd_name = commands[selected][0]
                        args = []
                        if cmd_name == "console":
                            port = Prompt.ask("Serial port", default=DEFAULT_PORT)
                            baud = DEFAULT_BAUD
                            exit_keys = ["ctrl-]", "ctrl-q", "ctrl-x"]
                            prefix_key = "ctrl-a"
                            pref_hint = f"prefix: {prefix_key} (x/q=exit, a=send, b=BREAK, h=help)"
                            conn_info = f"Connected to {port} @ {baud}  (exit: {', '.join(exit_keys)}, {pref_hint})"
                            menu_conn_info["info"] = conn_info
                            import subprocess
                            layout["status"].update(Text.from_markup("[yellow]Launching interactive console... (exit to return to menu)[/yellow]"))
                            live.update(layout)
                            try:
                                subprocess.run([sys.executable, sys.argv[0], "console", "--port", port])
                            except Exception as e:
                                status_text = Text.from_markup(f"[bold red]Error launching console: {e}[/bold red]")
                            continue
                        elif cmd_name == "log":
                            output_buffers.append(["[yellow]The 'log' command is interactive. Please run it directly in your terminal.[/yellow]"])
                            output_scrolls.append(0)
                            output_idx = len(output_buffers) - 1
                            console_text = Text.from_markup("[bold blue]Interactive console panel reserved. (Not implemented in UI)[/bold blue]")
                            status_text = Text.from_markup("[bold yellow]Interactive command. Not supported in UI.[/bold yellow]")
                            continue
                        elif cmd_name == "break":
                            port = Prompt.ask("Serial port", default=DEFAULT_PORT)
                            args = ["--port", port]
                        elif cmd_name == "run":
                            recipe = Prompt.ask("Recipe (netboot-debian, flash-bubt, flash-spi-manual, env-dump, env-save, reset)")
                            extra = Prompt.ask("Extra args (space-separated, optional)", default="")
                            args = ["--recipe", recipe]
                            if extra.strip():
                                args += extra.strip().split()
                        elif cmd_name == "kwboot":
                            port = Prompt.ask("Serial port", default=DEFAULT_PORT)
                            image = Prompt.ask("U-Boot image file (e.g., u-boot-uart.bin)")
                            extra = Prompt.ask("Extra kwboot args (optional)", default="")
                            args = ["--port", port, "--image", image]
                            if extra.strip():
                                args += ["--extra", extra]
                        from io import StringIO
                        from contextlib import redirect_stdout, redirect_stderr
                        output = StringIO()
                        try:
                            with redirect_stdout(output), redirect_stderr(output):
                                app([cmd_name] + args, standalone_mode=False)
                            status_text = Text.from_markup("[green]Command completed successfully.[/green]")
                        except SystemExit:
                            status_text = Text.from_markup("[green]Command exited.[/green]")
                        except Exception as e:
                            output.write(f"[ERROR] {e}\n")
                            status_text = Text.from_markup(f"[bold red]Error: {e}[/bold red]")
                        output_buffers.append([output.getvalue()])
                        output_scrolls.append(0)
                        output_idx = len(output_buffers) - 1
                        console_text = Text.from_markup("[dim]Interactive console will appear here when supported.[/dim]")
                    elif ch.lower() == 'q':
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# ---------- Interactive Menu ----------

@app.command("menu")
def cmd_menu():
    """Interactive menu to select and run commands."""
    commands = [
        ("list-ports", "List available serial devices."),
        ("console", "Interactive serial console with logging, exit keys, and prefix-escape."),
        ("console-ui", "Interactive split-pane UI for command selection and output."),
        ("log", "Non-interactive capture of serial output to a file."),
        ("break", "Spam CR to stop autoboot and land at U-Boot prompt."),
        ("run", "Run a U-Boot script or built-in recipe (flash/netboot/env/reset)."),
        ("kwboot", "UART boot a temporary U-Boot image."),
        ("doctor", "Environment checks (permissions, kwboot in PATH, ports)."),
        ("help", "Show this cheatsheet of commands and examples."),
    ]
    console.print(Panel("[bold cyan]MOCHAbin Tool Menu[/bold cyan]", title="Select a Command"))
    for idx, (cmd, desc) in enumerate(commands, 1):
        console.print(f"[bold]{idx}.[/bold] [green]{cmd}[/green] - {desc}")
    console.print()
    while True:
        try:
            choice = Prompt.ask("Enter command number (or 'q' to quit)")
            if choice.lower() == 'q':
                console.print("[bold]Exiting menu.[/bold]")
                return
            idx = int(choice) - 1
            if idx < 0 or idx >= len(commands):
                console.print("[red]Invalid selection.[/red]")
                continue
            cmd_name = commands[idx][0]
            # Gather args interactively for commands that require them
            args = []
            if cmd_name == "console":
                port = Prompt.ask("Serial port", default=DEFAULT_PORT)
                args = ["--port", port]
            elif cmd_name == "console-ui":
                import subprocess
                subprocess.run([sys.executable, sys.argv[0], "console-ui"])
                continue
            elif cmd_name == "log":
                port = Prompt.ask("Serial port", default=DEFAULT_PORT)
                outfile = Prompt.ask("Output log file", default="mochabin.log")
                seconds = Prompt.ask("Seconds to capture (0=forever)", default="0")
                args = ["--port", port, "--outfile", outfile, "--seconds", seconds]
            elif cmd_name == "break":
                port = Prompt.ask("Serial port", default=DEFAULT_PORT)
                args = ["--port", port]
            elif cmd_name == "run":
                recipe = Prompt.ask("Recipe (netboot-debian, flash-bubt, flash-spi-manual, env-dump, env-save, reset)")
                extra = Prompt.ask("Extra args (space-separated, optional)", default="")
                args = ["--recipe", recipe]
                if extra.strip():
                    args += extra.strip().split()
            elif cmd_name == "kwboot":
                port = Prompt.ask("Serial port", default=DEFAULT_PORT)
                image = Prompt.ask("U-Boot image file (e.g., u-boot-uart.bin)")
                extra = Prompt.ask("Extra kwboot args (optional)", default="")
                args = ["--port", port, "--image", image]
                if extra.strip():
                    args += ["--extra", extra]
            # doctor, list-ports, help need no args
            # Capture output
            output = StringIO()
            try:
                from contextlib import redirect_stdout, redirect_stderr
                with redirect_stdout(output), redirect_stderr(output):
                    app([cmd_name] + args, standalone_mode=False)
            except SystemExit:
                pass
            except Exception as e:
                output.write(f"[ERROR] {e}\n")
            text = Text.from_ansi(output.getvalue())
            console.print(Panel(text, title=f"Output: {cmd_name}", border_style="magenta"))
        except KeyboardInterrupt:
            console.print("[bold]Exiting menu.[/bold]")
            return


##FIRST RETRYBORNABLE

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.live import Live
from rich.prompt import Prompt
import subprocess
import sys
import os
import typer
import pexpect
import shlex
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.syntax import Syntax
from io import StringIO
import serial
from pathlib import Path
import time
from typing import Optional

# Default U-Boot prompt and timeout
DEFAULT_PROMPT = r"=> |# |Marvell>> "
DEFAULT_TIMEOUT = 5

# Console instance for Rich output
console = Console()

# Default serial port and baud rate
DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200

# Global Typer app for CLI
app = typer.Typer(help="MOCHAbin serial / U-Boot CLI.")

# --- Interactive Console UI Command ---
@app.command("console-ui")
def cmd_console_ui():
    """Interactive Rich UI: menu (top), output (middle), status (bottom)."""
    # Build command list from Typer app
    command_list = []
    for cmd in app.registered_commands:
        name = cmd.name
        if name not in ("console-ui", "help"):
            desc = cmd.help or ""
            command_list.append((name, desc))
    command_list.append(("Exit", "Exit the UI"))

    def draw_menu(selected_idx):
        menu = "\n".join([
            f"[bold green]> {name}[/bold green] - {desc}" if i == selected_idx else f"  {name} - {desc}"
            for i, (name, desc) in enumerate(command_list)
        ])
        return Panel(menu, title="[b]MOCHAbin Tool Menu", border_style="blue")

    def draw_status(status_text):
        return Panel(status_text, style="bold white on blue", title="Status", border_style="blue")

    selected_idx = 0
    output_text = ""
    status_text = "[dim]Ready. Use up/down/enter. Q to quit.[/dim]"

    with Live(console=console, refresh_per_second=10) as live:
        while True:
            layout = Layout()
            layout.split_column(
                Layout(draw_menu(selected_idx), name="menu", size=10),
                Layout(Panel(Align.left(Text.from_ansi(output_text)), title="Output", border_style="magenta"), name="output"),
                Layout(draw_status(status_text), name="status", size=3)
            )
            live.update(layout)
            key = Prompt.ask("[b]Select: [up/down/enter/q]", default="enter")
            if key.lower() in ("up", "k"):
                selected_idx = (selected_idx - 1) % len(command_list)
                status_text = f"[dim]Selected: {command_list[selected_idx][0]}[/dim]"
            elif key.lower() in ("down", "j"):
                selected_idx = (selected_idx + 1) % len(command_list)
                status_text = f"[dim]Selected: {command_list[selected_idx][0]}[/dim]"
            elif key.lower() in ("q", "quit"):
                status_text = "[yellow]Exiting...[/yellow]"
                live.update(layout)
                break
            elif key.lower() in ("enter", ""):
                name, _ = command_list[selected_idx]
                if name == "Exit":
                    status_text = "[yellow]Exiting...[/yellow]"
                    live.update(layout)
                    break
                # Prompt for arguments if needed
                cmd_obj = None
                for c in app.registered_commands:
                    if c.name == name:
                        cmd_obj = c
                        break
                params = []
                for param in (cmd_obj.params if cmd_obj else []):
                    if param.name in ("ctx", "self"):
                        continue
                    prompt_text = f"{param.name} ({param.help or param.type})"
                    default = param.default if param.default is not None else ""
                    value = Prompt.ask(prompt_text, default=str(default))
                    params.append(value)
                cmd = [sys.executable, os.path.abspath(__file__), name] + params
                try:
                    status_text = f"[cyan]Running: {name} {' '.join(params)}[/cyan]"
                    live.update(layout)
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    output_lines = []
                    for line in proc.stdout:
                        output_lines.append(line)
                        output_text = "".join(output_lines)
                        layout["output"].update(Panel(Align.left(Text.from_ansi(output_text)), title="Output", border_style="magenta"))
                        status_text = f"[cyan]Running: {name} ({len(output_lines)} lines)...[/cyan]"
                        layout["status"].update(draw_status(status_text))
                        live.update(layout)
                    proc.wait()
                    status_text = f"[green]Done: {name}[/green]"
                except Exception as e:
                    output_text = f"[red]Error: {e}[/red]"
                    status_text = f"[red]Error running {name}: {e}[/red]"
            else:
                status_text = "[yellow]Unknown key. Use up/down/enter/q.[/yellow]"

# ---------- Help / Cheatsheet ----------

def _build_command_table() -> Table:
    t = Table(title="MOCHAbin Tool – Commands", show_lines=False)
    t.add_column("Command", style="bold")
    t.add_column("What it does")
    t.add_row("list-ports", "List available serial devices.")
    t.add_row("console", "Interactive serial console with logging, exit keys, and prefix-escape.")
    t.add_row("log", "Non-interactive capture of serial output to a file.")
    t.add_row("break", "Spam CR to stop autoboot and land at U-Boot prompt.")
    t.add_row("run", "Run a U-Boot script or built-in recipe (flash/netboot/env/reset).")
    t.add_row("kwboot", "UART boot a temporary U-Boot image.")
    t.add_row("doctor", "Environment checks (permissions, kwboot in PATH, ports).")
    t.add_row("help", "Show this cheatsheet of commands and examples.")
    return t


def _build_examples_panel() -> Panel:
    examples = r"""# List ports
./mochabin_tool.py list-ports

# [Default: --port /dev/ttyUSB0]
# Console (quit with Ctrl-], Ctrl-Q, Ctrl-X). Prefix (Ctrl-A): x/q=exit, b=BREAK, a=send ^A
./mochabin_tool.py console
# Change prefix to Ctrl-] and only keep Ctrl-Q as direct exit:
./mochabin_tool.py console --exit-key ctrl-q --prefix-key ctrl-]
# Disable prefix:
./mochabin_tool.py console --no-prefix

# Stop autoboot and get to U-Boot prompt
./mochabin_tool.py break

# Flash U-Boot using 'bubt' from a USB stick (FAT, partition 1)
./mochabin_tool.py run --recipe flash-bubt flash-image.bin usb spi

# Manual SPI flash (fatload + sf)
./mochabin_tool.py run --recipe flash-spi-manual "usb 0:1" flash-image.bin

# UART boot a temporary U-Boot
./mochabin_tool.py kwboot --image u-boot-uart.bin --extra "-p -D 3"

# Netboot Debian installer via TFTP (server 192.168.1.10)
./mochabin_tool.py run --recipe netboot-debian 192.168.1.10 \
    debian-installer/arm64/linux \
    debian-installer/arm64/initrd.gz \
    "auto console=ttyS0,115200"

# Env + reset helpers
./mochabin_tool.py run --recipe env-dump
./mochabin_tool.py run --recipe env-save
./mochabin_tool.py run --recipe reset

# Capture boot log for 30s
./mochabin_tool.py log --outfile boot.log --seconds 30

# Doctor
./mochabin_tool.py doctor
"""
    return Panel(
        Syntax(examples, "bash", word_wrap=True),
        title="Quick Samples",
        border_style="cyan",
    )


def show_cheatsheet() -> None:
    console.print(_build_command_table())
    console.print(_build_examples_panel())


# ---------- App root (cheatsheet switch) ----------

@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    examples: bool = typer.Option(
        False,
        "--examples",
        "-x",
        help="Show a cheatsheet with commands and quick samples."
    ),
) -> None:
    """
    MOCHAbin serial / U-Boot CLI. Use --help for autogenerated help, or --examples for a quick cheatsheet.
    """
    if ctx.invoked_subcommand is None:
        if examples:
            show_cheatsheet()
        else:
            typer.echo(ctx.get_help())
            console.print("\n[dim]Tip: run with '--examples' or use the 'help' subcommand for quick samples.[/dim]")
        raise typer.Exit()


# ---------- Utilities ----------

def find_ports():
    return list(serial.tools.list_ports.comports())


def open_serial(port: str, baud: int = DEFAULT_BAUD, timeout: float = 0.2):
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=timeout,
        write_timeout=2.0,
    )
    return ser


def expect_spawn(ser, encoding: str = "utf-8"):
    child = fdpexpect.fdspawn(ser.fileno(), encoding=encoding, timeout=DEFAULT_TIMEOUT)
    child.logfile_read = None
    return child


CTRL_MAP = {
    "ctrl-a": 1,    # ^A
    "ctrl-]": 29,   # ^]
    "ctrl-q": 17,   # ^Q
    "ctrl-x": 24,   # ^X
    "ctrl-c": 3,    # ^C (captured because we set raw mode)
    "esc": 27,      # ESC
}


def normalize_exit_keys(keys):
    codes = set()
    for k in keys:
        k = k.lower().strip()
        if k in CTRL_MAP:
            codes.add(CTRL_MAP[k])
        elif k.startswith("0x"):
            codes.add(int(k, 16))
        elif k.isdigit():
            codes.add(int(k))
    return codes


@ app.command("help")
def cmd_help() -> None:
    """Show a concise cheatsheet with commands and quick samples."""
    show_cheatsheet()


def send_serial_break(ser, duration: float = 0.25):
    """
    Try to send a serial BREAK.
    Works on most USB-serial adapters; falls back to break_condition toggle.
    """
    try:
        ser.send_break(duration)
    except Exception:
        try:
            ser.break_condition = True
            time.sleep(duration)
        finally:
            ser.break_condition = False


def sendline(ser, line: str):
    if not line.endswith("\n"):
        line = line + "\n"
    ser.write(line.encode("utf-8"))
    ser.flush()


def flush_input(ser):
    try:
        ser.reset_input_buffer()
    except Exception:
        pass


def human_sleep(sec: float) -> None:
    # tiny spinner sleep
    end = time.time() + sec
    with Progress(SpinnerColumn(), TextColumn("[dim]waiting {task.description}"), transient=True) as progress:
        tid = progress.add_task(f"{sec:.1f}s", total=None)
        while time.time() < end:
            time.sleep(0.05)


# ---------- Console Commands ----------

@app.command("list-ports")
def cmd_list_ports() -> None:
    """List available serial ports."""
    ports = [p for p in find_ports() if getattr(p, 'device', None)]
    if not ports:
        console.print("[red]No serial ports found.[/red]")
        raise typer.Exit(1)
    table = Table(title="Serial ports")
    table.add_column("Device")
    table.add_column("Description")
    table.add_column("HWID")
    for p in ports:
        table.add_row(p.device, p.description or "?", p.hwid or "?")
    console.print(table)


@app.command("console")
def cmd_console(
    port: str = typer.Option(DEFAULT_PORT, help="Serial device"),
    baud: int = typer.Option(DEFAULT_BAUD, help="Baud rate"),
    log: Optional[Path] = typer.Option(None, help="Log all console output to file"),
    exit_key = typer.Option(
        ["ctrl-]", "ctrl-q", "ctrl-x"],
        help="Key(s) to exit console. Supported: ctrl-], ctrl-q, ctrl-x, ctrl-c, esc, or a byte value like 0x1d."
    ),
    prefix: bool = typer.Option(True, "--prefix/--no-prefix", help="Enable prefix-escape mode"),
    prefix_key: str = typer.Option("ctrl-a", help="Prefix key (e.g., ctrl-a, ctrl-], 0x1d)"),
) -> None:
    """
    Minimal interactive serial console.
    Exit with: Ctrl-], Ctrl-Q, or Ctrl-X (configurable via --exit-key).
    Prefix mode (like screen): press the prefix (default Ctrl-A), then:
      x/q = exit, a = send literal prefix, b = BREAK, h = help.
    """
    ser = open_serial(port, baud)
    exit_codes = normalize_exit_keys(exit_key)
    pretty_keys = ", ".join(exit_key)
    pref_code = next(iter(normalize_exit_keys([prefix_key])), None) if prefix else None
    pref_hint = f", prefix: {prefix_key} (x/q=exit, a=send, b=BREAK, h=help)" if pref_code is not None else ", prefix: disabled"
    console.print(f"[bold green]Connected[/bold green] to {port} @ {baud}  (exit: {pretty_keys}{pref_hint})")
    console.print("[dim]Tip: press Enter a couple times to surface the prompt[/dim]")
    console.print("[dim]Ctrl-D (EOF) will disconnect and exit the console.[/dim]")
    console.print()

    logfile = None
    if log:
        logfile = open(log, "ab")
        console.print(f"[cyan]Logging to[/cyan] {log}")

    try:
        # Non-canonical simple console
        import termios, tty, select
        fd_stdin = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd_stdin)
        tty.setraw(fd_stdin)
        try:
            prefix_armed = False
            while True:
                rlist, _, _ = select.select([fd_stdin, ser.fileno()], [], [])
                if ser.fileno() in rlist:
                    data = ser.read(ser.in_waiting or 1)
                    if data:
                        sys.stdout.buffer.write(data)
                        sys.stdout.flush()
                        if logfile:
                            logfile.write(data)
                            logfile.flush()
                if fd_stdin in rlist:
                    ch = os.read(fd_stdin, 1)
                    if not ch:
                        continue

                    # Prefix-escape state machine
                    if pref_code is not None:
                        if not prefix_armed and ch[0] == pref_code:
                            # Arm prefix, don't forward the prefix byte
                            prefix_armed = True
                            continue
                        elif prefix_armed:
                            # Handle prefix commands
                            c = ch.lower()
                            if c in (b'x', b'q'):
                                break
                            elif c == b'a' or ch[0] == pref_code:
                                # Send literal prefix to the serial peer
                                ser.write(bytes([pref_code]))
                            elif c == b'b':
                                send_serial_break(ser)
                            elif c == b'h':
                                # Local help (print on host; not sent to target)
                                sys.stdout.write("\r\n[local] prefix commands: x/q=exit, a=send-prefix, b=BREAK, h=help\r\n")
                                sys.stdout.flush()
                            else:
                                # Unknown prefix command: forward the char as-is
                                ser.write(ch)
                            prefix_armed = False
                            ser.flush()
                            continue

                    # Non-prefix path:
                    # Exit if the typed byte matches one of the configured exit keys
                    if ch[0] in exit_codes:
                        break
                    # Otherwise forward to the serial port
                    ser.write(ch)
                    ser.flush()
        finally:
            termios.tcsetattr(fd_stdin, termios.TCSADRAIN, old_settings)
    finally:
        if logfile:
            logfile.close()
        ser.close()
        console.print("\n[bold]Disconnected.[/bold]")


@app.command("log")
def cmd_log(
    port: str = typer.Option(DEFAULT_PORT, help="Serial device"),
    baud: int = typer.Option(DEFAULT_BAUD, help="Baud rate"),
    outfile: Path = typer.Option("mochabin.log", help="Output log file"),
    seconds: int = typer.Option(0, help="If >0, stop after N seconds"),
) -> None:
    """Non-interactive logging of serial output (good for capturing boot)."""
    ser = open_serial(port, baud)
    console.print(f"Capturing from {port} -> {outfile}")
    with open(outfile, "ab") as f, Live(transient=True) as live:
        start = time.time()
        try:
            while True:
                data = ser.read(ser.in_waiting or 1)
                if data:
                    f.write(data)
                    f.flush()
                if seconds and (time.time() - start) >= seconds:
                    break
        finally:
            ser.close()
    console.print("[green]Done.[/green]")


# ---------- U-Boot helpers ----------

def break_into_uboot(ser, child, timeout: int = 7, prompt: str = DEFAULT_PROMPT):
    """
    Stops autoboot and waits for the U-Boot prompt.
    Strategy: spam CRs during countdown; expect prompt.
    """
    console.print("[cyan]Breaking into U-Boot...[/cyan]")
    flush_input(ser)
    t0 = time.time()
    while (time.time() - t0) < timeout:
        ser.write(b'\r')
        ser.flush()
        try:
            child.expect(prompt, timeout=0.2)
            console.print("[green]U-Boot prompt detected.[/green]")
            return
        except Exception:
            pass
    # One more hard expect with a slightly longer wait
    child.expect(prompt, timeout=3)
    console.print("[green]U-Boot prompt detected.[/green]")


def run_uboot_lines(child, cmds, prompt: str = DEFAULT_PROMPT, echo: bool = True):
    for line in cmds:
        if not line.strip():
            continue
        if echo:
            console.print(f"[dim]> {line}[/dim]")
        child.sendline(line)
        child.expect(prompt, timeout=DEFAULT_TIMEOUT)


def recipe_netboot_debian(server: str, kernel_path: str, initrd_path: str, extra_args: str = ""):
    """
    Returns a list of U-Boot commands to TFTP (or HTTP*) boot Debian installer.
    Use http paths (http://server/path) or TFTP paths (just filenames).
    """
    is_http = kernel_path.startswith("http://") or kernel_path.startswith("https://")
    # U-Boot memory addresses: adjust if needed for your board
    kernel_addr = "0x02000000"
    initrd_addr = "0x06000000"
    fdt_addr = "${fdt_addr_r}"  # let U-Boot fill this if available

    cmds = [
        "setenv autoload no",
        "dhcp",  # get IP via DHCP
    ]
    if not is_http:
        cmds += [f"setenv serverip {server}"]

    load_kernel = (
        f"tftpboot {kernel_addr} {kernel_path}"
        if not is_http else
        # NOTE: HTTP flow is highly build-dependent; this placeholder tries EFI bootmgr path.
        f"setenv bootfile {kernel_path}; bootefi bootmgr"
    )

    load_initrd = f"tftpboot {initrd_addr} {initrd_path}" if not is_http else "echo Using HTTP boot manager..."

    bootargs = "console=ttyS0,115200 earlycon=uart,mmio32,0xf0512000"  # adjust UART if needed
    if extra_args:
        bootargs = f"{bootargs} {extra_args}"

    cmds += [
        f"setenv bootargs {bootargs}",
        load_kernel,
    ]
    if not is_http:
        cmds.append(load_initrd)
        # Try to use existing FDT if already in memory; otherwise ignore
        cmds.append(f"booti {kernel_addr} {initrd_addr} {fdt_addr}")
    return cmds


def recipe_flash_uboot_bubt(image: str, source: str = "usb", target: str = "spi"):
    """
    Use U-Boot's 'bubt' to burn image to SPI from usb/tftp/mmc.
    Example: bubt flash-image.bin spi usb
    """
    pre = []
    if source == "usb":
        pre = ["usb start", "ls usb 0:1"]
    elif source == "mmc":
        pre = ["mmc rescan", "ls mmc 0:1"]
    # For TFTP, ensure 'serverip' is set and image exists on TFTP root.
    return pre + [f"bubt {image} {target} {source}"]


def recipe_flash_uboot_spi_manual(image_dev: str = "usb 0:1", image_path: str = "flash-image.bin"):
    """
    Manual SPI flash procedure using fatload + sf.
    Adjust sizes/offsets for your image if needed.
    """
    load_addr = "${loadaddr}"
    return [
        "sf probe",                 # detect SPI NOR
        "usb start",                # or mmc rescan
        f"fatload {image_dev} {load_addr} {image_path}",
        "sf erase 0 +${filesize}",
        "sf write ${loadaddr} 0 ${filesize}",
        "sf read ${loadaddr} 0 ${filesize}",  # quick verify read-back (optional)
    ]


# ---------- Commands: U-Boot + recipes ----------

@app.command("break")
def cmd_break(
    port: str = typer.Option(DEFAULT_PORT),
    baud: int = typer.Option(DEFAULT_BAUD),
    prompt: str = typer.Option(DEFAULT_PROMPT, help="Regex for the U-Boot prompt"),
) -> None:
    """Stop autoboot and land at U-Boot prompt."""
    ser = open_serial(port, baud)
    child = expect_spawn(ser)
    try:
        break_into_uboot(ser, child, prompt=prompt)
        console.print("[bold green]Ready.[/bold green]")
    finally:
        ser.close()


@app.command("run")
def cmd_run(
    port: str = typer.Option(DEFAULT_PORT),
    baud: int = typer.Option(DEFAULT_BAUD),
    script: Optional[Path] = typer.Option(None, help="Text file of U-Boot commands to run"),
    recipe: Optional[str] = typer.Option(None, help="Built-in: netboot-debian | flash-bubt | flash-spi-manual | env-dump | env-save | reset"),
    args = typer.Argument(None, help="Extra args for recipe"),
    prompt: str = typer.Option(DEFAULT_PROMPT, help="Regex for the U-Boot prompt"),
    break_first: bool = typer.Option(True, help="Attempt to stop autoboot first"),
) -> None:
    """
    Run a U-Boot command script or a built-in recipe, and wait for the prompt after each line.
    """
    if not script and not recipe:
        console.print("[red]Provide --script or --recipe.[/red]")
        raise typer.Exit(2)

    ser = open_serial(port, baud)
    child = expect_spawn(ser)
    try:
        if break_first:
            break_into_uboot(ser, child, prompt=prompt)

        cmds = []
        if script:
            cmds = [ln.rstrip() for ln in script.read_text().splitlines()]
        else:
            r = recipe.lower()
            if r == "netboot-debian":
                if len(args) < 3:
                    console.print("[yellow]Usage:[/yellow] --recipe netboot-debian <server-ip> <kernel_path> <initrd_path> [extra_args...]")
                    raise typer.Exit(2)
                server, kernel_path, initrd_path, *extra = args
                extra_args = " ".join(extra) if extra else ""
                cmds = recipe_netboot_debian(server, kernel_path, initrd_path, extra_args)
            elif r == "flash-bubt":
                if len(args) < 1:
                    console.print("[yellow]Usage:[/yellow] --recipe flash-bubt <image> [source usb|tftp|mmc] [target spi|emmc]")
                    raise typer.Exit(2)
                image = args[0]
                source = args[1] if len(args) > 1 else "usb"
                target = args[2] if len(args) > 2 else "spi"
                cmds = recipe_flash_uboot_bubt(image=image, source=source, target=target)
            elif r == "flash-spi-manual":
                dev = args[0] if len(args) > 0 else "usb 0:1"
                path = args[1] if len(args) > 1 else "flash-image.bin"
                cmds = recipe_flash_uboot_spi_manual(image_dev=dev, image_path=path)
            elif r == "env-dump":
                cmds = ["printenv"]
            elif r == "env-save":
                cmds = ["saveenv"]
            elif r == "reset":
                cmds = ["reset"]
            else:
                console.print(f"[red]Unknown recipe:[/red] {recipe}")
                raise typer.Exit(2)

        run_uboot_lines(child, cmds, prompt=prompt)
        console.print("[green]Done.[/green]")
    finally:
        ser.close()


# ---------- kwboot (UART boot) ----------

@app.command("kwboot")
def cmd_kwboot(
    port: str = typer.Option(DEFAULT_PORT),
    image: Path = typer.Option(..., exists=True, help="U-Boot image for UART boot (e.g., u-boot-uart.bin)"),
    baud: int = typer.Option(DEFAULT_BAUD),
    extra: Optional[str] = typer.Option(None, help="Extra kwboot args, e.g. '-p -D 3'"),
) -> None:
    """
    Run kwboot to load a temporary U-Boot over UART, then attach an interactive console.
    """
    cmd = ["kwboot", "-b", str(image), "-B", str(baud), "-t", port]
    if extra:
        cmd = ["kwboot"] + shlex.split(extra) + ["-b", str(image), "-B", str(baud), "-t", port]

    console.print("[cyan]Launching kwboot:[/cyan] " + " ".join(shlex.quote(c) for c in cmd))
    console.print("[dim]Tip: if handshake fails, try adding '-p' or '-D 3' in --extra.[/dim]")
    try:
        # Stream kwboot until it exits (Ctrl-C to quit)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]kwboot failed:[/red] {e}")
        raise typer.Exit(e.returncode)


# ---------- Safety + Info ----------

@app.command("doctor")
def cmd_doctor() -> None:
    """Quick environment checks."""
    ok = True
    # Serial perms?
    me = os.geteuid() if hasattr(os, "geteuid") else -1
    if me == 0:
        msg = "[yellow]Running as root[/yellow] (ok, but consider udev groups)."
    else:
        msg = f"UID={me} (ensure your user is in 'dialout'/'uucp' on Linux)."
    console.print(msg)

    # kwboot available?
    if not shutil_which("kwboot"):
        ok = False
        console.print("[red]kwboot not found in PATH.[/red] Install it if you need UART boot.")
    else:
        console.print("[green]kwboot found.[/green]")

    # Python deps are present if we got here
    ports = find_ports()
    if ports:
        console.print(f"[green]{len(ports)}[/green] serial port(s) detected.")
    else:
        console.print("[yellow]No serial ports detected right now.[/yellow]")

    if ok:
        console.print("[bold green]Environment looks good.[/bold green]")
    else:
        raise typer.Exit(1)


def shutil_which(cmd: str) -> Optional[str]:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(p) / cmd
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


# ---------- Entry ----------

if __name__ == "__main__":
    app()
