# T-MAC FPGA Acceleration (DE10-Nano)

## Overview

FPGA accelerator for BitNet 1.58-bit GEMV using T-MAC lookup tables.
Replaces v1's 128-PE scalar ternary multiply with LUT-based computation.

**Target:** DE10-Nano (Cyclone V 5CSEBA6), 100 MHz
**Source:** `de10nano_bitnet/bitnet/chisel/src/main/scala/bitnet/TMac*.scala`

## Architecture

Two-phase operation per GEMV call:

1. **LUT Build** (~46 us): Read INT8 activations 3 at a time, compute 16-entry
   LUT per group, write to 32-bank BRAM. One-time cost per activation vector.

2. **GEMV Compute**: Stream T-MAC weights from DDR3 (nibbles + signs in separate
   arrays), 32 parallel lookup engines, 5-level pipelined adder tree, INT32 accumulator.

```
HPS ──Avalon-MM Slave──► TMacControlRegs ──► FSM
DDR3 ◄──Avalon-MM Master── TMacWeightStreamer / TMacActivationLoader / TMacResultWriter
TMacActivationBuffer ──► LutBuilder ──► LutBram ──► TMacComputeCore ──► ResultMem
```

## Weight Format (DDR3)

Separate nibble and sign arrays per weight matrix:

- **Nibble array** (NIB_BASE): 4 bits/group, 32 groups per 128-bit beat
- **Sign array** (SIGN_BASE): 1 bit/group, 128 groups per 128-bit beat

For K=6912 (n3=2304): 72 nibble beats + 18 sign beats = 90 beats/row

HPS driver must repack from software T-MAC format to this DDR3 layout.

## Register Map

| Offset | Name         | R/W | Description                           |
|--------|--------------|-----|---------------------------------------|
| 0x00   | CTRL         | W   | [0]=START, [1]=DDR3_MODE              |
| 0x04   | STATUS       | R   | [0]=BUSY, [1]=DONE                    |
| 0x08   | NIB_BASE     | R/W | DDR3 base for nibble array            |
| 0x0C   | DIM_M        | R/W | Output rows                           |
| 0x10   | DIM_K        | R/W | Input dimension (must be multiple of 3) |
| 0x14   | SIGN_BASE    | R/W | DDR3 base for sign array              |
| 0x18   | PERF_CYCLES  | R   | Clock cycles for last run             |
| 0x28   | ACT_DDR3_BASE| R/W | DDR3 base for INT8 activations        |
| 0x2C   | RES_DDR3_BASE| R/W | DDR3 base for INT32 results           |

## Performance (K=6912, M=6912, 100 MHz)

| Metric              | v1 (128-PE)  | T-MAC (P=32) |
|---------------------|--------------|--------------|
| DDR3 beats/row      | 108          | 90           |
| Effective cycles/row| 108          | 90           |
| GEMV time           | 7.46 ms      | 6.24 ms      |
| **Speedup**         | baseline     | **1.20x**    |

## Resource Estimate

| Resource | Used  | Available | %    |
|----------|-------|-----------|------|
| ALMs     | ~2,700| 41,910    | 6.4% |
| M10K     | ~100  | 553       | 18%  |
| DSPs     | 0     | 112       | 0%   |

## Software Driver

`biturbo_fpga.h` provides the HPS-side driver. Compile with `-DBT_FPGA`:

```bash
make fpga    # cross-compile for DE10-Nano ARM
```

### Integration Flow

```
bt_load_model() → tmac_repack() → bt_fpga_prepare_weights()
                                   (repacks all layers to FPGA tile format in CPU memory)

bt_forward() per layer:
  bt_fpga_load_layer()             → memcpy layer weights to DDR3 (cached)
  bitlinear_quantize()             → INT8 activations
  bt_fpga_upload_activations()     → memcpy to DDR3
  bt_fpga_gemv_dequant() × 7       → FPGA GEMV + dequant for Q,K,V,O,gate,up,down
```

### Environment Variables

- `BT_FPGA_DDR3_BASE`: DDR3 physical base (default `0x30000000`)
- `BT_FPGA_DDR3_SPAN`: DDR3 region size (default `0x02000000` = 32 MB)

### Weight Repack (Software → FPGA)

Nibble: software even g = high nibble → FPGA even engine = low nibble (swapped)
Sign: software bit g%8 → FPGA bit (g%32) in 32-bit chunk at (tile%4)×32

## Build

```bash
# FPGA RTL (Chisel → SystemVerilog)
cd de10nano_bitnet/bitnet/chisel
sbt "runMain bitnet.TMacAccelMain"    # generates generated/TMacAccelerator.sv
sbt test                               # runs all unit tests

# Software driver (cross-compile for DE10-Nano)
cd biturbo.c
make fpga                              # builds biturbo_fpga binary
```
