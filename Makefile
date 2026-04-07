CC      = cc
CFLAGS  = -std=gnu99 -O2 -Wall -Wextra -Wpedantic -DNDEBUG
LDFLAGS = -lm

# ARM NEON (auto-detected on macOS Apple Silicon)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
  CFLAGS += -mcpu=apple-m1
endif
ifeq ($(UNAME_M),aarch64)
  CFLAGS += -march=armv8-a+fp+simd
endif

TARGET  = biturbo
SRCS    = main.c biturbo.c
OBJS    = $(SRCS:.c=.o)

.PHONY: all clean debug fpga

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c biturbo.h
	$(CC) $(CFLAGS) -c -o $@ $<

debug: CFLAGS = -std=gnu99 -O0 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: clean $(TARGET)

# FPGA build: cross-compile for DE10-Nano ARM with T-MAC accelerator
FPGA_CC     ?= arm-linux-gnueabihf-gcc
FPGA_CFLAGS  = -std=gnu99 -O2 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -DBT_FPGA -DNDEBUG
FPGA_LDFLAGS = -lm

fpga:
	$(FPGA_CC) $(FPGA_CFLAGS) -o biturbo_fpga main.c biturbo.c $(FPGA_LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET) biturbo_fpga
