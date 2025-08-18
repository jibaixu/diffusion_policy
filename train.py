"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
def debug_on():
    import os
    import sys
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # os.environ['http_proxy'] = 'http://100.84.172.223:7897'
    # os.environ['https_proxy'] = 'http://100.84.172.223:7897'
    # os.environ['ftp_proxy'] = 'http://100.84.172.223:7897'
    sys.argv = [
        "train.py",
        "--config-dir=diffusion_policy/config",
        "--config-name=train_diffusion_transformer_hybrid_image_language_workspace.yaml",
        "task.dataset.zarr_path=data/AllTasks-v3/zarr_xarm6_traj700_multiview",
        "hydra.run.dir=outputs/train/AllTasks-v3/xarm6",
        "training.seed=42",
        "training.device=cuda:0",
        "training.num_epochs=300",
        "training.checkpoint_every=20",
        "training.resume=False",
        "dataloader.batch_size=64",
        "dataloader.num_workers=4",
        "dataloader.persistent_workers=False",
        "val_dataloader.batch_size=64",
        "val_dataloader.num_workers=4",
        "val_dataloader.persistent_workers=False",
        "logging.mode=offline",    # ["online", "offline", "disabled"]
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
