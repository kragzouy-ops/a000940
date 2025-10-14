# mbrola-interface

A tiny Python wrapper for [MBROLA] with optional text→`.pho` using eSpeak-NG.

## Install (from source)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install .
```

## CLI Usage

```bash
mbrola-interface --voice /usr/share/mbrola/en1/en1 --demo --out demo.wav --play
```

Or from text:

```bash
mbrola-interface --voice /usr/share/mbrola/en1/en1   --text "Hello from MBROLA" --espeak-voice mb-en1   --out hello.wav --dump-pho hello.pho
```

## GUI

```bash
mbrola-interface --gui
```

The GUI lets you browse for voices, type text, or load existing `.pho` files without remembering CLI flags. Optional fields expose the same parameters as the CLI (volume/time/freq ratios, dump `.pho`, playback, etc.).

The interface follows an Apple Human Interface Guidelines look and feel:

- "MBROLA Studio" workspace with soft neutral surfaces (`#F5F5F7`) and elevated white cards for each task group.
- SF Pro-inspired typography, accent/secondary button treatments, and rounded tab navigation.
- Friendly dialog copy and status messages that mirror macOS tone.

You can toggle between Text, Phoneme, and Demo tabs, tweak synthesis parameters, and optionally auto-play the rendered file.

## Requirements
- `mbrola` installed + at least one voice (e.g., `mbrola-en1`, `mbrola-fr1`)
- `espeak-ng` (for text→`.pho`)

[MBROLA]: https://github.com/numediart/MBROLA (binary distribution site varies by distro)
