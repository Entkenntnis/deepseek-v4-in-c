# SPDX-License-Identifier: Apache-2.0
# deepseek-v4-in-c -- usage: make tui | make test | make clean

CC       ?= cc
CFLAGS   ?= -O3 -std=gnu99 -Wall -Wextra -Wpointer-arith -Wshadow -Wvla \
            -Wno-unused-parameter -march=native -fopenmp -pthread -ffp-contract=off
INCS     := -Iinclude -Iinclude/dsv4 -Ithird_party -Isrc/core -Isrc/io \
            -Isrc/cache -Isrc/model -Isrc/tokenizer -Isrc/prefill
LDFLAGS  ?= -lm -fopenmp -pthread
DEPFLAGS := -MMD -MP

BIN := bin
OBJ := build/obj

ENGINE_SRC := $(wildcard src/core/*.c src/io/*.c src/cache/*.c src/model/*.c \
                        src/tokenizer/*.c src/prefill/*.c)
ENGINE_OBJ := $(patsubst src/%.c,$(OBJ)/src/%.o,$(ENGINE_SRC))
SERVE_OBJ  := $(OBJ)/src/cli/dsv4_serve.o

TESTS    := quant matmul ops hc rope attn compress indexer layer
TEST_BIN := $(addprefix $(BIN)/test_,$(TESTS))
TEST_OBJ := $(addprefix $(OBJ)/tests/unit/test_,$(addsuffix .o,$(TESTS)))

.PHONY: test tui clean
.SECONDARY: $(TEST_OBJ)

test: $(TEST_BIN)
	@set -e; for t in $(TEST_BIN); do echo; ./$$t; done

tui: $(BIN)/dsv4-serve
	python3 tui/dsv4_tui.py --budget auto --ctx 8192 --gen 1024

$(BIN)/dsv4-serve: $(SERVE_OBJ) $(ENGINE_OBJ)
	@mkdir -p $(BIN)
	$(CC) $^ -o $@ $(LDFLAGS)

$(BIN)/test_%: $(OBJ)/tests/unit/test_%.o $(ENGINE_OBJ)
	@mkdir -p $(BIN)
	$(CC) $^ -o $@ $(LDFLAGS)

$(OBJ)/src/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCS) -c $< -o $@

$(OBJ)/tests/%.o: tests/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCS) -c $< -o $@

clean:
	rm -rf $(BIN) build

-include $(ENGINE_OBJ:.o=.d) $(TEST_OBJ:.o=.d) $(SERVE_OBJ:.o=.d)
