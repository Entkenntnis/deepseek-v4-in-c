/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_portable_io.h - Linux I/O shims.
 *
 * The engine targets Linux. O_DIRECT is an open() flag and posix_fadvise is
 * available, so the old Darwin shims are gone. dsv4_set_direct() remains as the
 * one interface the readers call: on Linux, O_DIRECT was already applied by
 * open(), so there is nothing left to do on the descriptor.
 */
#ifndef DSV4_PORTABLE_IO_H
#define DSV4_PORTABLE_IO_H

#include <fcntl.h>

static inline int dsv4_set_direct(int fd) { (void)fd; return 0; }

#endif /* DSV4_PORTABLE_IO_H */
