# Board_B_HAT_mikroBUS
Date: 2025-10-14

**What this is:** A minimal, openable KiCad project skeleton with the key connectors placed:
- 40-pin Pi-style header (JHAT/J1)
- mikroBUS sockets (J2–J5) on Board B
- Generic MOCHAbin I/O header (JMOCHA) on Board A
- Power input (JVIN) and an optional USB bridge header on Board A

**Next steps for you:**
1. Wire nets according to `pin_map.csv` (in the starter pack) using labels and wires.
2. Add level shifters (PCA9306, SN74AXC8T245), power (buck + 3V3), polyfuses, TVS, and the I2C ADC (ADS1015).
3. Pick footprints to match your preferred connectors (PinSocket vs PinHeader, stack heights).
4. Create the PCB (`.kicad_pcb`) from the schematic and place keep-outs matching Raspberry Pi HAT outline.

> This skeleton intentionally uses only standard KiCad connector symbols to maximize portability. 
> You can place custom IC symbols from vendor libs, or add them to `project_symbols.kicad_sym` later.


**Update:** Initial `.kicad_pcb` added with board outline and connector footprints placed. Verify positions and align to official specs before manufacturing.


**Mechanical precision:** HAT outline 65.0 × 56.5 mm; mounting-hole centers at (3.5,3.5), (61.5,3.5), (3.5,52.5), (61.5,52.5) mm. mikroBUS row spacing = **22.86 mm** (900 mil); pin pitch 2.54 mm.
