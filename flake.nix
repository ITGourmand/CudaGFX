{
  description = "Multi-platform C++ development environment (native Linux + Windows cross-compilation)";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        # mingw-w64 cross toolchain, used by `make windows`
        mingwPkgs = pkgs.pkgsCross.mingwW64;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.gnumake
            pkgs.cmake
            pkgs.gdb
            mingwPkgs.stdenv.cc
            mingwPkgs.buildPackages.binutils
            # Test tooling (tests/run_tests.sh):
            pkgs.glslang    # glslangValidator: validates the generated GLSL when shaderc is not available
            pkgs.shellcheck # lint of the shell scripts
          ];
        };
      });
}
