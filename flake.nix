{
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pydeps = pkgs.python310.withPackages
          (ps: with ps; [
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
          ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [ pydeps pkgs.yosys pkgs.parallel ];
          packages = with pkgs; [ gtkwave ];
        };

        packages.docker = pkgs.dockerTools.buildLayeredImage
          {
            name = "hsd-project";
            contents = with pkgs; [
              coreutils
              pydeps
              vim
              yosys
              (
                let src = ./.; in
                pkgs.runCommand "code" { } ''
                  mkdir -p $out
                  cp -rv ${src} $out/code
                ''
              )
            ];

            config = {
              Cmd = [ "${pkgs.bash}/bin/bash" ];
              WorkingDir = "code/";
            };
          };
      }
    );
}
