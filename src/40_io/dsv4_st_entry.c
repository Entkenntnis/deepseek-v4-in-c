/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_st_entry.c - parse one tensor entry of a safetensors header. */
#include <stdio.h>
#include <string.h>

#include "dsv4_st.h"
#include "dsv4_st_int.h"

int dsv4_st_parse_entry(DSV4Scan *sc, const char *path, const char *name,
                        int64_t base, DSV4Tensor *t)
{
    if (!dsv4_st_lit(sc, '{')) {
        fprintf(stderr, "dsv4_st: %s entry is not an object\n", path);
        return -1;
    }

    int have_dt = 0, have_off = 0;
    int64_t o0 = 0, o1 = 0;

    int f1 = 1;
    for (;;) {
        dsv4_st_ws(sc);
        if (sc->p < sc->end && *sc->p == '}') { sc->p++; break; }
        if (!f1 && !dsv4_st_lit(sc, ',')) return -1;
        f1 = 0;
        char key[64]; size_t klen;
        if (!dsv4_st_str(sc, key, sizeof key, &klen)) return -1;
        if (!dsv4_st_lit(sc, ':')) return -1;

        if (!strcmp(key, "dtype")) {
            char dv[32]; size_t dl;
            if (!dsv4_st_str(sc, dv, sizeof dv, &dl)) return -1;
            t->dtype = dsv4_st_dtype_of(dv, dl);
            if (t->dtype == DSV4_DT_UNKNOWN) {
                fprintf(stderr, "dsv4_st: %s: unsupported dtype '%s' on %s\n",
                        path, dv, name);
                return -1;
            }
            have_dt = 1;
        } else if (!strcmp(key, "shape")) {
            if (!dsv4_st_lit(sc, '[')) return -1;
            dsv4_st_ws(sc);
            if (sc->p < sc->end && *sc->p == ']') sc->p++;       /* scalar: [] */
            else for (;;) {
                int64_t d;
                if (!dsv4_st_i64(sc, &d)) return -1;
                if (t->ndim < 4) t->shape[t->ndim] = d;
                else {
                    fprintf(stderr, "dsv4_st: %s has rank > 4\n", name);
                    return -1;
                }
                t->ndim++;
                dsv4_st_ws(sc);
                if (dsv4_st_lit(sc, ',')) continue;
                if (dsv4_st_lit(sc, ']')) break;
                return -1;
            }
        } else if (!strcmp(key, "data_offsets")) {
            if (!dsv4_st_lit(sc, '[')) return -1;
            if (!dsv4_st_i64(sc, &o0)) return -1;
            if (!dsv4_st_lit(sc, ',')) return -1;
            if (!dsv4_st_i64(sc, &o1)) return -1;
            if (!dsv4_st_lit(sc, ']')) return -1;
            have_off = 1;
        } else {
            if (!dsv4_st_skip_value(sc)) return -1;
        }
    }

    if (!have_dt || !have_off) {
        fprintf(stderr, "dsv4_st: %s: %s is missing dtype or data_offsets\n",
                path, name);
        return -1;
    }

    /* The byte span must equal elements times element size. A mismatch
     * means shape and data disagree, and every later read would be
     * silently misaligned. Refuse rather than load it. */
    t->off    = base + o0;
    t->nbytes = o1 - o0;
    const int64_t want = dsv4_st_numel(t) * dsv4_st_elemsize(t->dtype);
    if (t->nbytes != want) {
        fprintf(stderr, "dsv4_st: %s: %s spans %lld bytes but shape implies %lld\n",
                path, name, (long long)t->nbytes, (long long)want);
        return -1;
    }
    return 0;
}
