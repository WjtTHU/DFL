{
  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-parts,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
    ...
  } @ inputs:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      perSystem = {
        config,
        self',
        inputs',
        pkgs,
        system,
        ...
      }: let
        pkgs = import inputs.nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            # cudaSupport = true;
          };
        };

        python = pkgs.python312;
        pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
        projectVersion = pyproject.project.version or "0.0.1";

        workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ./.;
        };

        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };
        editableOverlay = workspace.mkEditablePyprojectOverlay {
          root = "./.";
        };
        customOverlay = self: super: {
          torch = super.torch.overrideAttrs (old: {
            nativeBuildInputs =
              (old.nativeBuildInputs or [])
              ++ [
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
                pkgs.cudaPackages.libcusparse
                pkgs.cudaPackages.libcusparse_lt
                pkgs.cudaPackages.libcufile
                pkgs.cudaPackages.libnvshmem
                pkgs.cudaPackages.nccl
              ];
            autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or []) ++ ["libcuda.so.1"];
          });
          nvidia-cufile-cu12 = super.nvidia-cufile-cu12.overrideAttrs (old: {
            nativeBuildInputs =
              (old.nativeBuildInputs or [])
              ++ [
                pkgs.rdma-core
              ];
          });
          nvidia-nvshmem-cu12 = super.nvidia-nvshmem-cu12.overrideAttrs (old: {
            nativeBuildInputs =
              old.nativeBuildInputs
              ++ [
                pkgs.openmpi
                pkgs.pmix
                pkgs.ucx
                pkgs.libfabric
                pkgs.rdma-core
              ];
          });
          nvidia-cusparse-cu12 = super.nvidia-cusparse-cu12.overrideAttrs (old: {
            nativeBuildInputs =
              (old.nativeBuildInputs or [])
              ++ [
                pkgs.cudaPackages.libnvjitlink
              ];
          });
          nvidia-cusolver-cu12 = super.nvidia-cusolver-cu12.overrideAttrs (old: {
            nativeBuildInputs =
              (old.nativeBuildInputs or [])
              ++ [
                pkgs.cudaPackages.libnvjitlink
                pkgs.cudaPackages.libcusparse
                pkgs.cudaPackages.libcublas
              ];
          });
        };

        pythonSets =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
          (
            pkgs.lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
              customOverlay
            ]
          );

        mkWipDockerImage = {
          name,
          command,
          extraAppDirs ? [],
          exposedPorts ? {},
          volumes ? {},
        }: let
          appDirs =
            [
              "tmp"
              "app"
              "app/config"
              "app/scripts"
              "app/checkpoints"
            ]
            ++ extraAppDirs;
        in
          pkgs.dockerTools.buildLayeredImage {
            inherit name;
            tag = projectVersion;
            created = "now";

            contents = [
              self'.packages.pythonDeps
              self'.packages.cudaRuntimeDeps
              self'.packages.baseDeps
            ];

            extraCommands = ''
              ${pkgs.lib.concatMapStringsSep "\n" (dir: "mkdir -p ${dir}") appDirs}

              # Copy application code
              cp -r ${./.}/src app/
              cp -r ${./.}/scripts app/
              cp -r ${./.}/sql app/
              cp ${./.}/pyproject.toml app/
            '';

            config =
              {
                WorkingDir = "/app";
                Env = [
                  "PYTHONPATH=/app/src"
                  "PATH=${self'.packages.default}/bin:${self'.packages.baseDeps}/bin:/bin"
                  "LD_LIBRARY_PATH=${self'.packages.cudaRuntimeDeps}/lib:${self'.packages.cudaRuntimeDeps}/lib64:${self'.packages.baseDeps}/lib:${self'.packages.baseDeps}/lib64"
                ];
                Cmd = command;
              }
              // pkgs.lib.optionalAttrs (exposedPorts != {}) {
                ExposedPorts = exposedPorts;
              }
              // pkgs.lib.optionalAttrs (volumes != {}) {
                Volumes = volumes;
              };
          };
      in {
        apps.package-logs = {
          type = "app";
          program = "${
            pkgs.writeShellApplication {
              name = "package-logs";
              runtimeInputs = [
                python
                pkgs.git
                pkgs.p7zip
              ];
              text = ''
                repo="''${WIP_ANALYSIS_REPO:-$PWD}"
                exec ${python}/bin/python "$repo/scripts/package_logs.py" --repo "$repo" "$@"
              '';
            }
          }/bin/package-logs";
        };

        # Dependency Packs
        packages.baseDeps = pkgs.buildEnv {
          name = "basic-dependencies";
          paths = [
            pkgs.coreutils
            pkgs.bash
            pkgs.libpq
          ];
        };
        packages.pythonDeps = pkgs.buildEnv {
          name = "python-packages";
          paths = [self'.packages.default];
        };
        packages.cudaRuntimeDeps = pkgs.buildEnv {
          name = "cuda-runtime";
          ignoreCollisions = true;
          paths = [
            pkgs.cudaPackages.cuda_cudart
            pkgs.cudaPackages.libcublas
            pkgs.cudaPackages.libcusolver
            pkgs.cudaPackages.libcusparse
            pkgs.cudaPackages.libcufft
            pkgs.cudaPackages.libcurand
            pkgs.cudaPackages.libnvjitlink
            pkgs.cudaPackages.cudnn
          ];
        };

        # Dev Shells
        devShells.default = let
          pythonSet = pythonSets.overrideScope editableOverlay;
          virtualenv = pythonSet.mkVirtualEnv pyproject.project.name workspace.deps.all;
        in
          pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
              pkgs.prek
              pkgs.ruff
              pkgs.ty
              pkgs.perl
              pkgs.pgcli
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.cuda_nvcc
            ];

            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              unset PYTHONPATH
              ln -sfn ${virtualenv} .venv
            '';
          };

        # Products
        packages.default = pythonSets.mkVirtualEnv pyproject.project.name workspace.deps.default;

        packages.dockerTrain = mkWipDockerImage {
          name = "wip-analysis-train";
          extraAppDirs = ["app/experiments"];
          command = [
            "${self'.packages.default}/bin/python"
            "scripts/training/train.py"
          ];
          volumes = {
            "/app/checkpoints" = {};
            "/app/experiments" = {};
            "/app/config" = {};
          };
        };

        packages.dockerApi = mkWipDockerImage {
          name = "wip-analysis-api";
          exposedPorts = {
            "8724/tcp" = {};
          };
          command = [
            "${self'.packages.default}/bin/python"
            "scripts/api/start_api.py"
          ];
        };
      };
    };
}
