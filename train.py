"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
def debug_on():
    import os
    import sys
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sys.argv = [
        "train.py",
        "--config-dir=diffusion_policy/config",
        "--config-name=train_diffusion_transformer_hybrid_image_language_workspace.yaml",
        "training.seed=42",
        "training.device=cuda:0",
        "hydra.run.dir=outputs/train/MultiTasks-v1/xarm",
    ]
debug_on()


import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
