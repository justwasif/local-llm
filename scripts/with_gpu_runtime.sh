#!/usr/bin/env bash
# Launch a command with the Nix NVIDIA driver and C++ runtime visible to pip wheels.
set -euo pipefail

gpu_lib="/run/opengl-driver/lib"
cpp_lib="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
# Nix zlib path (found via find)
zlib_lib="/nix/store/8icpg7vrz95c6ap3mznmlmg7h0l2av1w-zlib-1.3.1/lib"
# Nix numactl path
numa_lib="/nix/store/87kd8nb5bws9x8r3k66hbcryr3xxvwxn-numactl-2.0.18/lib"

export LD_LIBRARY_PATH="${gpu_lib}:${cpp_lib}:${zlib_lib}:${numa_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$@"
