{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    pynng-flake.url = "github:afermg/pynng";
    pynng-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        nahualPkg = pkgs.python3.pkgs.callPackage (inputs.nahual-flake + "/nix/nahual.nix") {
          pynng = inputs.pynng-flake.packages.${system}.pynng;
        };
        python_with_pkgs = pkgs.python3.withPackages (pp: [
          nahualPkg
          pp.torch
          pp.torchvision
          pp.einops
          pp.timm
          pp.fvcore
          pp.numpy
          pp.pillow
        ]);
        runServer = pkgs.writeScriptBin "nahual-channelsformer" ''
          #!${pkgs.bash}/bin/bash
          exec ${python_with_pkgs}/bin/python ${self}/server.py \
            "''${1:-tcp://0.0.0.0:5555}"
        '';
        channelsformerApp = {
          type = "app";
          program = "${runServer}/bin/nahual-channelsformer";
        };
      in
        with pkgs; rec {
          packages = pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            oci-image = import ./nix/oci-image.nix {
              inherit pkgs;
              name = "channelsformer";
              title = "Nahual ChannelSFormer";
              description = "ChannelSFormer feature extraction served through Nahual";
              source = "https://github.com/afermg/ChannelSFormer";
              revision = self.rev or self.dirtyRev or "unknown";
              server = runServer;
              entrypoint = channelsformerApp.program;
            };
          };
          inherit python_with_pkgs;
          scripts.runServer = runServer;
          apps = rec {
            channelsformer = channelsformerApp;
            default = channelsformer;
          };
          devShells.default = mkShell {
            packages = [
              python_with_pkgs
              pkgs.cudaPackages.cudatoolkit
              python3Packages.tifffile
              python3Packages.scikit-image
              python3Packages.scikit-learn
              python3Packages.pyyaml
            ];
            shellHook = ''
              export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
            '';
          };
        }
    );
}
