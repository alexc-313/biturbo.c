CC      = cc
CFLAGS  = -std=c99 -O2 -Wall -Wextra -Wpedantic -DNDEBUG
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

.PHONY: all clean debug

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c biturbo.h
	$(CC) $(CFLAGS) -c -o $@ $<

debug: CFLAGS = -std=c99 -O0 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: clean $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
