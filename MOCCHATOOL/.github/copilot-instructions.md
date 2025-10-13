# Copilot Instructions

- **Project Role**: `mochabin_tool.py` is a Typer-based CLI for interacting with MOCHAbin hardware over serial; everything routes through this single module.
- **Entrypoint**: All commands are registered on the global `app = typer.Typer(...)` and invoked via `python mochabin_tool.py <command>` or by making the script executable.
- **Hardware Loop**: Serial access flows through `open_serial()` (pyserial) and `expect_spawn()` (pexpect). `run_uboot_lines()` sends each command and waits for `DEFAULT_PROMPT`; adjust prompts carefully when touching recipes or expect-based logic.
- **Console UX**: `cmd_console` implements raw tty handling with prefix-escape behavior; any key handling change must respect `normalize_exit_keys()` and the prefix state machine.
- **Recipes**: `cmd_run` dispatches to helpers such as `recipe_netboot_debian`, `recipe_flash_uboot_bubt`, and `recipe_flash_uboot_spi_manual`. Extend by adding a new `recipe_*` function and enumerating it inside the `elif` chain; maintain informative usage messages.
- **Autoboot Flow**: `break_into_uboot()` spams carriage returns and waits for the prompt; tuning timeouts impacts every recipe. Use the shared constants (`DEFAULT_TIMEOUT`, `DEFAULT_PROMPT`) rather than hard-coded numbers.
- **Logging**: `cmd_log` writes binary output for reliability. When adding file writes elsewhere, mimic the binary mode and flush behavior to avoid partial captures.
- **kwboot Integration**: `cmd_kwboot` shells out via `subprocess.run`; pass arguments through `shlex.split(extra)`. Stay mindful of shell quoting and do not replace with `shell=True`.
- **Environment Checks**: `cmd_doctor` reports serial access and `kwboot` availability using the local helper `shutil_which`; reuse it instead of importing `shutil.which` to keep behavior consistent.
- **Error Handling**: Most commands exit using `typer.Exit`; follow the existing pattern (print Rich-styled message, then raise with a status code) for new error paths.
- **Dependency Setup**: Run `install.sh` to create `.venv` and install `typer[all]`, `pyserial`, `pexpect`, and `rich`. Activate the venv before running commands in development.
- **Manual Verification**: There is no automated test suite; validate changes by running the specific Typer command (e.g., `./mochabin_tool.py --examples`, `./mochabin_tool.py run --recipe env-dump`). When hardware is unavailable, consider mocking serial ports when adding logic that does not require real I/O.
- **Coding Style**: Keep functions top-level in `mochabin_tool.py`, prefer small helpers for serial flows, and add brief Rich console cues to guide users during long-running steps.
