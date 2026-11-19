# deepseek-v4-in-c

Upstream: https://github.com/ronak-create/deepseek-v4-in-c

This file is for humans, never touch or change by agent.

---

Optimised version for my system: Intel Core Ultra 7 258V, 32 GB RAM, 1 TB NVMe

Setup:

- https://github.com/ronak-create/deepseek-v4-in-c#running-a-model -> download Deepseek V4 Flash from huggingface and run the 2 python scripts to create the packs. (~160GiB)
- Python and https://github.com/Entkenntnis/deepseek-v4-in-c/blob/main/tui/requirements.txt for the TUI
- Basic gcc toosl (make, gcc, vulkan, openmp, ... see Makefile)

Main entry point is `make tui` and start chatting.

Run `make test` for some sanity checks.

## Licence

Apache-2.0. Portions derived from `kimi-k3-in-c`, also Apache-2.0.
