{
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python310
            python310Packages.amaranth
            python310Packages.numpy
            yosys
          ];
          packages = with pkgs; [ gtkwave ];
        };
      }
    );
}
