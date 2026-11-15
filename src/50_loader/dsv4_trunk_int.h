/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trunk_int.h - seams between the trunk files. Not public. */
#ifndef DSV4_TRUNK_INT_H
#define DSV4_TRUNK_INT_H

#include "dsv4.h"
#include "dsv4_trunk.h"

/* Parse trunk.json into tr->lay, then resolve plan_layer()'s request order to
 * manifest indices once per layer. */
int dsv4_trunk_load_manifest(DSV4Trunk *tr, const char *dir);
int dsv4_trunk_build_maps(DSV4Trunk *tr, const DSV4Cfg *c);

/* Read one layer's run into dst, which must hold slot_bytes. */
int dsv4_trunk_read_run(DSV4Trunk *tr, int L, unsigned char *dst);

#endif /* DSV4_TRUNK_INT_H */
