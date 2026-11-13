# SPDX-License-Identifier: Apache-2.0
# deepseek-v4-in-c

CC      ?= cc
CFLAGS  ?= -O3 -std=gnu99 -Wall -Wextra -Wpointer-arith -Wshadow -Wvla \
           -Wno-unused-parameter -march=native -fopenmp -pthread -ffp-contract=off
# -ffp-contract=off is not an optimisation choice. The scalar, OpenMP and AVX2
# paths must produce bit-identical results, so that a performance change can
# never quietly become an accuracy change. Fusing a multiply and an add changes
# the rounding, so it stays off.
INCS    := -Iinclude -Iinclude/dsv4 -Ithird_party -Isrc/core -Isrc/io \
           -Isrc/cache -Isrc/model -Isrc/tokenizer
LDFLAGS ?= -lm -fopenmp -pthread

BUILD := build
BIN   := bin

.PHONY: all test bench bench-decode cfg-fixtures st-fixtures tok-fixtures fixtures clean help tui

# EVERY binary depends on EVERY engine source and header.
#
# The per-target rules below list only the sources each one compiles, which is
# accurate but not sufficient: editing src/cache/dsv4_cache.c did NOT rebuild
# bin/test_cache, so a gate silently kept testing the previous build and
# reported a failure that had already been instrumented away. Twice.
#
# Rebuilding twenty small binaries on any header change costs a few seconds and
# removes a whole class of phantom results, so it is not a trade worth tuning.
ENGINE := $(wildcard src/core/*.c src/core/*.h src/io/*.c src/io/*.h \
                     src/cache/*.c src/cache/*.h src/model/*.c src/model/*.h \
                     src/tokenizer/*.c src/tokenizer/*.h \
                     include/dsv4/*.h third_party/*.h)
ALLBIN := $(BIN)/test_pro $(BIN)/dsv4 $(BIN)/dsv4-serve $(BIN)/test_cfg $(BIN)/test_st $(BIN)/test_bind $(BIN)/test_quant $(BIN)/test_matmul $(BIN)/test_ops $(BIN)/test_hc $(BIN)/test_rope $(BIN)/test_attn $(BIN)/test_compress $(BIN)/test_indexer $(BIN)/test_layer $(BIN)/test_cache $(BIN)/test_bindmem $(BIN)/test_trunk $(BIN)/test_tok

all: $(ALLBIN)

$(ALLBIN): $(ENGINE)


$(BIN)/test_cfg: tests/unit/test_cfg.c include/dsv4/dsv4.h include/dsv4/dsv4_cfg.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) $< -o $@ $(LDFLAGS)

$(BIN)/test_st: tests/unit/test_st.c src/io/dsv4_st.c src/io/dsv4_st.h \
                src/io/dsv4_portable_io.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_st.c src/io/dsv4_st.c -o $@ $(LDFLAGS)

# Regenerate fixtures. Both are derived from the released checkpoint, so they are
# only regenerated when it changes; the outputs are committed.
cfg-fixtures:
	python3 tools/make_cfg_fixtures.py
st-fixtures:
	python3 tools/make_st_fixture.py tests/fixtures/st
fixtures: cfg-fixtures st-fixtures

$(BIN)/test_bind: tests/unit/test_bind.c src/model/dsv4_bind.c src/model/dsv4_bind.h                   src/io/dsv4_st.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_bind.c src/model/dsv4_bind.c 	      src/io/dsv4_st.c -o $@ $(LDFLAGS)

$(BIN)/test_quant: tests/unit/test_quant.c src/core/dsv4_quant.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_quant.c -o $@ $(LDFLAGS)

$(BIN)/test_matmul: tests/unit/test_matmul.c src/core/dsv4_matmul.c                     src/core/dsv4_quant.h include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_matmul.c src/core/dsv4_matmul.c 	      -o $@ $(LDFLAGS)

$(BIN)/test_ops: tests/unit/test_ops.c src/core/dsv4_ops.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_ops.c src/core/dsv4_ops.c -o $@ $(LDFLAGS)

$(BIN)/test_hc: tests/unit/test_hc.c src/core/dsv4_hc.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_hc.c src/core/dsv4_hc.c -o $@ $(LDFLAGS)

$(BIN)/test_rope: tests/unit/test_rope.c src/core/dsv4_rope.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_rope.c src/core/dsv4_rope.c -o $@ $(LDFLAGS)

$(BIN)/test_attn: tests/unit/test_attn.c src/core/dsv4_attn.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_attn.c src/core/dsv4_attn.c -o $@ $(LDFLAGS)

$(BIN)/test_compress: tests/unit/test_compress.c src/core/dsv4_compress.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_compress.c src/core/dsv4_compress.c -o $@ $(LDFLAGS)

$(BIN)/test_indexer: tests/unit/test_indexer.c src/core/dsv4_indexer.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_indexer.c src/core/dsv4_indexer.c -o $@ $(LDFLAGS)

DSV4_CORE := src/core/dsv4_ops.c src/core/dsv4_matmul.c src/core/dsv4_hc.c              src/core/dsv4_rope.c src/core/dsv4_attn.c              src/core/dsv4_compress.c src/core/dsv4_indexer.c              src/core/dsv4_threads.c

$(BIN)/test_layer: tests/unit/test_layer.c src/model/dsv4_layer.c $(DSV4_CORE)                    src/model/dsv4_layer.h include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_layer.c src/model/dsv4_layer.c 	      $(DSV4_CORE) -o $@ $(LDFLAGS)

$(BIN)/test_cache: tests/unit/test_cache.c src/cache/dsv4_cache.c src/io/dsv4_st.c                    src/core/dsv4_threads.c src/cache/dsv4_cache.h include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) -Isrc/cache tests/unit/test_cache.c 	      src/cache/dsv4_cache.c src/io/dsv4_st.c src/core/dsv4_threads.c -o $@ $(LDFLAGS)

$(BIN)/test_bindmem: tests/unit/test_bindmem.c src/model/dsv4_bind.c src/io/dsv4_st.c                      src/model/dsv4_bind.h include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_bindmem.c src/model/dsv4_bind.c 	      src/io/dsv4_st.c -o $@ $(LDFLAGS)

$(BIN)/test_trunk: tests/unit/test_trunk.c src/io/dsv4_trunk.c src/io/dsv4_st.c                    src/model/dsv4_bind.c src/io/dsv4_trunk.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_trunk.c src/io/dsv4_trunk.c 	      src/io/dsv4_st.c src/model/dsv4_bind.c -o $@ $(LDFLAGS)

$(BIN)/test_tok: tests/unit/test_tok.c src/tokenizer/dsv4_tok.c                  src/tokenizer/dsv4_tok.h third_party/dsv4_unicode.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_tok.c src/tokenizer/dsv4_tok.c 	      -o $@ $(LDFLAGS)

tok-fixtures:
	python3 tools/pack_tokenizer.py $(DSV4_MODEL) tests/fixtures/tok/tokenizer.bin
	~/venv-cuda/bin/python tools/emit_tok_fixture.py $(DSV4_MODEL) tests/fixtures/tok

DSV4_ENGINE := src/io/dsv4_st.c src/io/dsv4_trunk.c src/cache/dsv4_cache.c                src/model/dsv4_bind.c src/model/dsv4_layer.c                src/tokenizer/dsv4_tok.c $(DSV4_CORE)

$(BIN)/dsv4: src/cli/dsv4_run.c $(DSV4_ENGINE)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) src/cli/dsv4_run.c $(DSV4_ENGINE) -o $@ $(LDFLAGS)

# The TUI backend. Same engine, one JSON event per line on stdout. Kept separate
# from dsv4_run.c so the one-shot CLI stays exactly what the gates and docs
# describe.
$(BIN)/dsv4-serve: src/cli/dsv4_serve.c $(DSV4_ENGINE)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) src/cli/dsv4_serve.c $(DSV4_ENGINE) -o $@ $(LDFLAGS)

$(BIN)/test_pro: tests/unit/test_pro.c $(DSV4_ENGINE)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_pro.c $(DSV4_ENGINE) -o $@ $(LDFLAGS)

test: $(BIN)/test_pro $(BIN)/test_cfg $(BIN)/test_st $(BIN)/test_bind $(BIN)/test_quant $(BIN)/test_matmul $(BIN)/test_ops $(BIN)/test_hc $(BIN)/test_rope $(BIN)/test_attn $(BIN)/test_compress $(BIN)/test_indexer $(BIN)/test_layer $(BIN)/test_cache $(BIN)/test_bindmem $(BIN)/test_trunk $(BIN)/test_tok
	@./$(BIN)/test_cfg
	@echo
	@./$(BIN)/test_st
	@echo
	@./$(BIN)/test_bind
	@echo
	@./$(BIN)/test_quant
	@echo
	@./$(BIN)/test_matmul
	@echo
	@./$(BIN)/test_ops
	@echo
	@./$(BIN)/test_hc
	@echo
	@./$(BIN)/test_rope
	@echo
	@./$(BIN)/test_attn
	@echo
	@./$(BIN)/test_compress
	@echo
	@./$(BIN)/test_indexer
	@echo
	@./$(BIN)/test_layer
	@echo
	@./$(BIN)/test_cache
	@echo
	@./$(BIN)/test_bindmem
	@echo
	@./$(BIN)/test_trunk
	@echo
	@./$(BIN)/test_tok
	@echo
	@./$(BIN)/test_pro

# ------------------------------------------------------------- BENCHES ------
# The README tells you to run these, so something has to build them. They link
# the same objects with the same CFLAGS as the engine -- a bench built with
# different flags measures a different program.
BENCHBIN := $(BIN)/matmul_bw $(BIN)/cache_bw $(BIN)/fp8_decode $(BIN)/fp4_f32 $(BIN)/matmul_variants $(BIN)/ctx_kernels $(BIN)/bind_resolve
bench: $(BENCHBIN)

# Context-scaling kernels: top-k, indexer, sparse attention and mHC, old vs new,
# on a synthetic KV cache. No checkpoint and no disk, so it can run while the
# machine is busy with I/O.
$(BIN)/ctx_kernels: bench/ctx_kernels.c src/core/dsv4_indexer.c \
                    src/core/dsv4_attn.c src/core/dsv4_hc.c src/core/dsv4_dot.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) bench/ctx_kernels.c src/core/dsv4_indexer.c \
	      src/core/dsv4_attn.c src/core/dsv4_hc.c -o $@ $(LDFLAGS)

# Name resolution cost on the per-token bind path, real manifest, no reads.
$(BIN)/bind_resolve: bench/bind_resolve.c src/io/dsv4_trunk.c src/io/dsv4_st.c \
                     src/model/dsv4_bind.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) bench/bind_resolve.c src/io/dsv4_trunk.c \
	      src/io/dsv4_st.c src/model/dsv4_bind.c -o $@ $(LDFLAGS)

$(BIN)/matmul_bw: bench/matmul_bw.c src/core/dsv4_matmul.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) bench/matmul_bw.c src/core/dsv4_matmul.c 	      -o $@ $(LDFLAGS)

$(BIN)/matmul_variants: bench/matmul_variants.c bench/matmul_experiments.c src/core/dsv4_matmul.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) -DDSV4_MATMUL_EXPERIMENTS bench/matmul_variants.c bench/matmul_experiments.c src/core/dsv4_matmul.c -o $@ $(LDFLAGS)

$(BIN)/fp4_f32: bench/fp4_f32.c src/core/dsv4_matmul.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) bench/fp4_f32.c src/core/dsv4_matmul.c 	      -o $@ $(LDFLAGS)

$(BIN)/cache_bw: bench/cache_bw.c src/cache/dsv4_cache.c src/io/dsv4_st.c src/core/dsv4_threads.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) bench/cache_bw.c src/cache/dsv4_cache.c 	      src/io/dsv4_st.c src/core/dsv4_threads.c -o $@ $(LDFLAGS)

$(BIN)/fp8_decode: bench/fp8_decode.c src/core/dsv4_matmul.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) bench/fp8_decode.c src/core/dsv4_matmul.c -o $@ $(LDFLAGS)

# The decode-rate baseline from BENCHMARK.md. Three runs at ~60 s each is the
# price of a number this machine can be held to.
bench-decode: $(BIN)/dsv4
	python3 bench/bench_decode.py --reps 3

# ---------------------------------------------------------------- TUI --------
tui: $(BIN)/dsv4-serve
	python3 tui/dsv4_tui.py --budget auto --ctx 8192 --gen 1024

clean:
	rm -rf $(BUILD) $(BIN)

help:
	@echo "deepseek-v4-in-c -- DeepSeek-V4 in C99, streamed off NVMe"
	@echo ""
	@echo "  make               build everything"
	@echo "  make test          the gate ladder (the trunk gate needs"
	@echo "                     DSV4_MODEL and DSV4_TRUNK)"
	@echo "  make bench         matmul_bw, cache_bw, ctx_kernels"
	@echo "  make bench-decode  the decode-rate baseline (BENCHMARK.md)"
	@echo "  make tui           build the streaming backend and launch the Python TUI"
	@echo "  make clean"
	@echo ""
	@echo "fixtures (only needed if you change a kernel or a format):"
	@echo "  make fixtures      all of the below"
	@echo "  make cfg-fixtures  bad-config fixtures from the released config"
	@echo "  make st-fixtures   a synthetic safetensors shard"
	@echo "  make tok-fixtures  tokenizer parity cases"
	@echo ""
	@echo "running a model (see README.md):"
	@echo "  python3 tools/pack_trunk.py     <model_dir> <trunk_dir>"
	@echo "  python3 tools/pack_tokenizer.py <model_dir> <tok.bin>"
	@echo "  ./bin/dsv4 <model_dir> --trunk <dir> --tok <file> \\"
	@echo "             --prompt TEXT --gen 25 --budget 16"
	@echo ""
	@echo "live TUI (chat + timings/cache/disk/RSS while it runs):"
	@echo "  python3 tui/dsv4_tui.py --demo      # no checkpoint needed"
	@echo "  python3 tui/dsv4_tui.py             # real model, defaults in tui/README.md"
	@echo ""
	@echo "  --budget must exceed one pass's working set (Flash 3.21 GB)"
	@echo "  plus the trunk (6.27 GB), or the expert cache cannot hit at all."
