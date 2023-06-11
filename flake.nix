{
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            (python310.withPackages (ps: with ps; [
              amaranth
              numpy
              pytorch
              pytorch-lightning
              (safetensors.overrideAttrs (prev: rec {
                version = "0.3.1";
                src = pkgs.fetchFromGitHub {
                  inherit (prev.src) owner repo;
                  rev = "v${version}";
                  hash = "sha256-RoIBD+zBKVzXE8OpI8GR371YPxceR4P8B9T1/AHc9vA=";
                };
                patches = [ ];
                cargoDeps = prev.cargoDeps.overrideAttrs (_: { inherit patches; });
              }))
              torchmetrics
              torchvision
            ]))
            yosys
          ];
          packages = with pkgs; [ gtkwave ];
        };
      }
    );
}
