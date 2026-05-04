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

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        # nahual recipe sourced from upstream flake input; built locally so
        # we can supply pynng (not in nixos-24.11).
        nahualPkg = pkgs.python3.pkgs.callPackage (inputs.nahual-flake + "/nix/nahual.nix") {
          pynng = inputs.pynng-flake.packages.${system}.pynng;
        };
      in
      with pkgs;
      rec {
        apps.default =
          let
            python_with_pkgs = python3.withPackages (pp: [
              nahualPkg
              pp.torch
              pp.torchvision
              pp.einops
              pp.timm
              pp.fvcore
              pp.numpy
              pp.pillow
            ]);
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/channelsformer.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };

        devShells = {
          default =
            let
              python_with_pkgs = python3.withPackages (pp: [
                nahualPkg
                pp.torch
                pp.torchvision
                pp.einops
                pp.timm
                pp.fvcore
                pp.tifffile
                pp.scikit-image
                pp.scikit-learn
                pp.numpy
                pp.pyyaml
              ]);
            in
            mkShell {
              packages = [
                python_with_pkgs
                pkgs.cudaPackages.cudatoolkit
              ];
              shellHook = ''
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
              '';
            };
        };
      }
    );
}
