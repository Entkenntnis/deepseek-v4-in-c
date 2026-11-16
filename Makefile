# SPDX-License-Identifier: Apache-2.0
# deepseek-v4-in-c -- usage: make tui | make test | make bench | make e2e | make clean

CC       ?= cc
CFLAGS   ?= -O3 -std=gnu99 -Wall -Wextra -Wpointer-arith -Wshadow -Wvla \
            -Wno-unused-parameter -march=native -fopenmp -pthread -ffp-contract=off -flto=auto
INCS     := -Iinclude -Iinclude/dsv4 -Ithird_party -Ibuild \
            -Isrc/00_base -Isrc/10_compute -Isrc/20_runtime -Isrc/30_tokenizer \
            -Isrc/40_io -Isrc/50_loader -Isrc/60_cache -Isrc/70_decode \
            -Isrc/80_prefill
LDFLAGS  ?= -lm -fopenmp -pthread
# The engine owns its expert slots in Vulkan memory (src/60_cache). `bench`
# does not link the cache, so it stays free of libvulkan.
VK_LDFLAGS ?= -lvulkan
DEPFLAGS := -MMD -MP

# Plain `make` has always run the test suite; keep that behaviour.
.DEFAULT_GOAL := test

BIN := bin
OBJ := build/obj

COMPUTE_SRC := $(wildcard src/10_compute/*.c)
RUNTIME_SRC := $(wildcard src/20_runtime/*.c)
ENGINE_SRC  := $(filter-out src/90_cli/%,$(wildcard src/*/*.c))
ENGINE_OBJ  := $(patsubst src/%.c,$(OBJ)/src/%.o,$(ENGINE_SRC))
CLI_SRC     := $(wildcard src/90_cli/*.c)
SERVE_OBJ   := $(patsubst src/%.c,$(OBJ)/src/%.o,$(CLI_SRC))

COMPUTE_OBJ := $(patsubst src/%.c,$(OBJ)/src/%.o,$(COMPUTE_SRC))
RUNTIME_OBJ := $(patsubst src/%.c,$(OBJ)/src/%.o,$(RUNTIME_SRC))
# vk_expert has its own main and needs libvulkan; keep it out of `bin/bench`
# so the CPU-only benchmark stays free of both.
VKX_SRC     := bench/vk_expert.c
BENCH_SRC   := $(filter-out $(VKX_SRC),$(wildcard bench/*.c))
BENCH_OBJ   := $(patsubst bench/%.c,$(OBJ)/bench/%.o,$(BENCH_SRC))
VKX_OBJ     := $(patsubst bench/%.c,$(OBJ)/bench/%.o,$(VKX_SRC))
VK_SPV      := build/expert_fp4.spv
VK_SPV_H    := build/expert_fp4_spv.h

TESTS    := quant matmul ops hc rope attn compress indexer layer
TEST_BIN := $(addprefix $(BIN)/test_,$(TESTS))
TEST_OBJ := $(addprefix $(OBJ)/tests/unit/test_,$(addsuffix .o,$(TESTS)))
REF_OBJ  := $(OBJ)/tests/unit/ref_kernels.o

.PHONY: test serve tui bench vk-expert e2e clean
.SECONDARY: $(TEST_OBJ) $(REF_OBJ)

test: $(TEST_BIN)
	@set -e; for t in $(TEST_BIN); do echo; ./$$t; done

serve: $(BIN)/dsv4-serve

tui: serve
	python3 tui/dsv4_tui.py --budget auto --ctx 8192 --gen 1024

bench: $(BIN)/bench
	./$(BIN)/bench

e2e: $(BIN)/dsv4-serve
	python3 bench/e2e.py

$(BIN)/dsv4-serve: $(SERVE_OBJ) $(ENGINE_OBJ)
	@mkdir -p $(BIN)
	$(CC) $^ -o $@ $(LDFLAGS) $(VK_LDFLAGS)

$(BIN)/bench: $(BENCH_OBJ) $(COMPUTE_OBJ) $(RUNTIME_OBJ)
	@mkdir -p $(BIN)
	$(CC) $^ -o $@ $(LDFLAGS) $(VK_LDFLAGS)

# One shader, two consumers: the standalone vk_expert bench loads the .spv,
# and the engine embeds the same words through this generated header.
$(VK_SPV): src/10_compute/shaders/expert_fp4.comp
	@mkdir -p build
	glslc -O --target-env=vulkan1.3 -o $@ $<

$(VK_SPV_H): $(VK_SPV) tools/spv_to_header.py
	python3 tools/spv_to_header.py $< $@ expert_fp4_spv

$(OBJ)/src/60_cache/dsv4_expert_vk.o: $(VK_SPV_H)

$(BIN)/vk_expert: $(VKX_OBJ) $(OBJ)/bench/fixtures.o $(COMPUTE_OBJ) \
                  $(RUNTIME_OBJ) $(VK_SPV)
	@mkdir -p $(BIN)
	$(CC) $(VKX_OBJ) $(OBJ)/bench/fixtures.o $(COMPUTE_OBJ) $(RUNTIME_OBJ) \
	      -o $@ $(LDFLAGS) $(VK_LDFLAGS)

vk-expert: $(BIN)/vk_expert
	./$(BIN)/vk_expert

$(BIN)/test_%: $(OBJ)/tests/unit/test_%.o $(REF_OBJ) $(ENGINE_OBJ)
	@mkdir -p $(BIN)
	$(CC) $^ -o $@ $(LDFLAGS) $(VK_LDFLAGS)

$(OBJ)/src/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCS) -c $< -o $@

$(OBJ)/tests/%.o: tests/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCS) -c $< -o $@

$(OBJ)/bench/%.o: bench/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCS) -c $< -o $@

clean:
	rm -rf $(BIN) build

-include $(ENGINE_OBJ:.o=.d) $(TEST_OBJ:.o=.d) $(REF_OBJ:.o=.d) \
         $(SERVE_OBJ:.o=.d) $(BENCH_OBJ:.o=.d) $(VKX_OBJ:.o=.d)
