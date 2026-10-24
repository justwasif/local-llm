{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "localllm-env";

  buildInputs = with pkgs; [
    numactl
    zlib
    stdenv.cc.cc.lib
    pkg-config
    icu
    cudaPackages.cudatoolkit
    cudaPackages.cuda_nvcc
    python312
    python312Packages.pip
    python312Packages.setuptools
    python312Packages.wheel
  ];

  shellHook = ''
    export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
    export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.numactl}/lib:${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.icu}/lib"
    
    echo "LocalLLM Nix Shell active"
  '';
}
