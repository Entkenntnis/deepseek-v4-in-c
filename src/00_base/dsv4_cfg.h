/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cfg.h - the fixed DeepSeek-V4-Flash configuration.
 *
 * This build is tuned for exactly one checkpoint, so there is nothing to parse:
 * the values that used to come from config.json live in dsv4_cfg.c. A different
 * architecture or checkpoint is a different build, not a runtime option.
 *
 * Keeping the config as constants removes a whole failure mode. The old reader
 * had to reject absent fields because a plausible default (compress_ratios
 * empty, hc_mult 1) produced a DIFFERENT model that still emitted fluent text.
 * A compiled-in table cannot be half-read.
 */
#ifndef DSV4_CFG_H
#define DSV4_CFG_H

#include "dsv4.h"

/* Fill `c` with the DeepSeek-V4-Flash geometry. compress_ratios points at a
 * static table owned by dsv4_cfg.c; the caller must not free it. */
void dsv4_cfg_init(DSV4Cfg *c);

#endif /* DSV4_CFG_H */
