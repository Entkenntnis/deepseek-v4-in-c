# SPDX-License-Identifier: Apache-2.0
# deepseek-v4-in-c -- usage: make tui | make demo | make test | make e2e | make prefill | make clean

CC       ?= cc
CFLAGS   ?= -O3 -std=gnu99 -Wall -Wextra -Wpointer-arith -Wshadow -Wvla \
            -Wno-unused-parameter -march=native -fopenmp -pthread -ffp-contract=off -flto=auto
INCS     := -Ibuild \
            -Isrc/00_base -Isrc/10_compute -Isrc/20_runtime \
            -Isrc/40_io -Isrc/50_loader -Isrc/60_cache -Isrc/70_decode \
            -Isrc/80_prefill
LDFLAGS  ?= -lm -fopenmp -pthread
VK_LDFLAGS ?= -lvulkan
DEPFLAGS := -MMD -MP

# Plain `make` has always run the test suite; keep that behaviour.
.DEFAULT_GOAL := test

BIN := bin
OBJ := build/obj

ENGINE_SRC  := $(filter-out src/90_cli/%,$(wildcard src/*/*.c))
ENGINE_OBJ  := $(patsubst src/%.c,$(OBJ)/src/%.o,$(ENGINE_SRC))
CLI_SRC     := $(wildcard src/90_cli/*.c)
SERVE_OBJ   := $(patsubst src/%.c,$(OBJ)/src/%.o,$(CLI_SRC))

TESTS    := quant matmul ops hc rope attn compress indexer layer kv
TEST_BIN := $(addprefix $(BIN)/test_,$(TESTS))
TEST_OBJ := $(addprefix $(OBJ)/tests/unit/test_,$(addsuffix .o,$(TESTS)))
REF_OBJ  := $(OBJ)/tests/unit/ref_kernels.o

# Prefill GEMM shaders, embedded into dsv4_prefill_vk_init.o.
VK_SPV   := build/gemm_fp8.spv build/gemm_bf16.spv build/gemm_fp4.spv \
            build/swiglu.spv
VK_SPV_H := $(patsubst build/%.spv,build/%_spv.h,$(VK_SPV))

.PHONY: test serve tui demo e2e prefill kv clean
.SECONDARY: $(TEST_OBJ) $(REF_OBJ)

test: $(TEST_BIN)
	@set -e; for t in $(TEST_BIN); do echo; ./$$t; done

serve: $(BIN)/dsv4-serve

tui: serve
	python3 tui/dsv4_tui.py

demo:
	python3 tui/dsv4_tui.py --demo

e2e: $(BIN)/dsv4-serve
	python3 bench/e2e.py

prefill: $(BIN)/dsv4-serve
	python3 bench/prefill.py

# Precompute the KV state for the /tools preamble. Always rebuild: the
# preamble or model may have changed, and a stale file is cheap to replace.
kv: $(BIN)/dsv4-serve
	python3 tools/build_kv_prefix.py

$(BIN)/dsv4-serve: $(SERVE_OBJ) $(ENGINE_OBJ)
	@mkdir -p $(BIN)
	$(CC) $^ -o $@ $(LDFLAGS) $(VK_LDFLAGS)

# Prefill shaders: compile once, embed the words so the binary needs no .spv.
$(VK_SPV): build/%.spv: src/10_compute/shaders/%.comp
	@mkdir -p build
	glslc -O --target-env=vulkan1.3 -o $@ $<

$(VK_SPV_H): build/%_spv.h: build/%.spv tools/spv_to_header.py
	python3 tools/spv_to_header.py $< $@ $*_spv

$(OBJ)/src/80_prefill/dsv4_prefill_vk_init.o: $(VK_SPV_H)

$(BIN)/test_%: $(OBJ)/tests/unit/test_%.o $(REF_OBJ) $(ENGINE_OBJ)
	@mkdir -p $(BIN)
	$(CC) $^ -o $@ $(LDFLAGS) $(VK_LDFLAGS)

$(OBJ)/src/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCS) -c $< -o $@

$(OBJ)/tests/%.o: tests/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCS) -c $< -o $@

clean:
	rm -rf $(BIN) build

-include $(ENGINE_OBJ:.o=.d) $(TEST_OBJ:.o=.d) $(REF_OBJ:.o=.d) \
         $(SERVE_OBJ:.o=.d)
