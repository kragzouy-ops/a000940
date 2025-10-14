# MBROLA Interface AI Coding Instructions

## Project Overview
This is a Python wrapper for the MBROLA speech synthesis system. The codebase is intentionally minimal—a single `mbrola_interface.py` file that acts as both a library and CLI tool.

## Architecture & Key Components

### Single-File Design
- **Everything lives in `mbrola_interface.py`**: Library class, CLI, utilities, and error handling
- **Zero Python dependencies**: Uses only stdlib (`subprocess`, `tempfile`, `pathlib`, etc.)
- **External binary dependencies**: `mbrola` (required) and `espeak-ng` (optional for text→phoneme conversion)

### Core Class: `MBROLA`
```python
tts = MBROLA(voice_path="/usr/share/mbrola/en1/en1")
tts.synthesize_text("Hello", "out.wav", espeak_voice="mb-en1")  # Text input
tts.synthesize_pho(pho_content, "out.wav")  # Direct phoneme input
```

### Voice Path Resolution Logic
The `_resolve_voice_file()` function handles flexible voice paths:
- File path: `/usr/share/mbrola/en1/en1` → use directly
- Directory: `/usr/share/mbrola/en1/` → auto-discover voice file inside
- Fallback priority: same basename as dir > single extensionless file > longest filename

## Critical Workflows

### Testing Voice Functionality
```bash
# System dependency check
mbrola -i /usr/share/mbrola/en1/en1 - -  # Should show voice info

# Quick demo (tests full pipeline)
mbrola-interface --voice /usr/share/mbrola/en1/en1 --demo --out test.wav --play
```

### Development Setup
```bash
./install_debian.sh  # Installs system deps + creates .venv + pip install .
# OR manual:
python3 -m venv .venv && . .venv/bin/activate && pip install .
```

## Project-Specific Patterns

### Error Handling Philosophy
- Custom `MBROLAError` for all domain errors
- Binary validation with `_ensure_executable()` - checks PATH + file permissions
- Subprocess errors include full command + stderr for debugging

### Audio Format Detection
Output format determined by file extension only: `.wav`, `.au`, `.aiff`, `.raw`

### Phoneme (.pho) Format
MBROLA phoneme files have specific syntax (see `examples/hello.pho`):
```
_ 150        # silence, 150ms
h 70         # phoneme 'h', 70ms
o 120  0 110  60 110  # phoneme 'o', 120ms, pitch contour points
```

### CLI Design Pattern
- Mutually exclusive input groups: `--text` + `--espeak-voice` OR `--pho-file` OR `--demo`
- `--demo` mode uses default text "Hello from MBROLA" with `mb-en1` voice
- All MBROLA parameters (volume, time ratio, freq ratio) exposed as CLI flags

## Integration Points

### System Dependencies
- **MBROLA binary**: Core synthesis engine, distro packages: `mbrola`, `mbrola-en1`, `mbrola-fr1`
- **eSpeak-NG**: Text→phoneme conversion, package: `espeak-ng`
- **Audio playback**: Best-effort via `ffplay`/`aplay`/`paplay`/`afplay`/`play`

### Voice Database Locations
Standard paths: `/usr/share/mbrola/{voice}/` (e.g., `/usr/share/mbrola/en1/en1`)

## Development Notes

### When Adding Features
- Keep zero-dependency principle - use stdlib only
- Add CLI arguments to `_build_arg_parser()` if exposing new functionality
- Use `_ensure_executable()` for any new binary dependencies
- Error messages should include command + stderr for subprocess failures

### Testing Strategy
- Use `--demo` mode for quick integration tests
- Test voice path resolution with various input formats
- Verify audio output with `--play` flag (requires audio system)

### Common Gotchas
- Voice paths can be files OR directories - always use `_resolve_voice_file()`
- MBROLA expects UTF-8 encoded phoneme input via stdin
- espeak-ng voices follow `mb-{lang}` pattern (e.g., `mb-en1`, `mb-fr1`)