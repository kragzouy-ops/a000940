# KiCad Skeletons for mikroBUS ⇄ MOCHAbin Two-Part Adapter
Date: 2025-10-14

This archive contains two openable KiCad projects:
- `Board_A_BaseBridge/Board_A_BaseBridge.kicad_pro` — Base/Bridge (MOCHAbin-facing, level shifting + power).
- `Board_B_HAT_mikroBUS/Board_B_HAT_mikroBUS.kicad_pro` — HAT-shaped dual mikroBUS adapter (two sockets).

They include the essential connectors as symbols and comments with mapping defaults. 
Wire the nets following the `pin_map.csv` you already have, then add the active components and footprints.

**Tip:** Use 11 mm stacking header height between boards to match typical HAT spacing.
