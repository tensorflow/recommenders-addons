# Compile flags for linux

SHOBJ_CFLAGS ?= -W -Wall -fno-common -g -ggdb -std=c99 -O2 -march=native
SHOBJ_LDFLAGS ?= -shared

.SUFFIXES: .c .so .xo .o

all: bpv2_hmaccum_cmd.so

.c.xo:
	$(CC) -I. $(CFLAGS) $(SHOBJ_CFLAGS) -fPIC -c $< -o $@

bpv2_hmaccum_cmd.xo: redismodule.h

bpv2_hmaccum_cmd.so: bpv2_hmaccum_cmd.xo
	$(LD) -o $@ $< $(SHOBJ_LDFLAGS) $(LIBS) -lc

clean:
	rm -rf *.xo *.so