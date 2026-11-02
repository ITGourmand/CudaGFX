
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

        # Linux graphics runtime dependencies (bgfx / Vulkan / OpenGL)
        graphicsPackages = pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
          pkgs.vulkan-loader
          pkgs.libGL
          pkgs.libx11
          pkgs.libxext
        ];

        # Shared library search paths
        graphicsLibraryPath = pkgs.lib.makeLibraryPath graphicsPackages;

      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.gnumake
            pkgs.cmake
            pkgs.gdb

            # Windows cross-compilation
            mingwPkgs.stdenv.cc
            mingwPkgs.buildPackages.binutils

            # Test tooling
            pkgs.glslang
            pkgs.shellcheck
          ] ++ graphicsPackages;

          # Graphics runtime environment (Linux only)
          shellHook = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:${graphicsLibraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        };
      });
}
