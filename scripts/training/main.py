import os
import glob
import subprocess
import tarfile
import wget
import copy
from omegaconf import OmegaConf, open_dict
import pickle

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging, exp_manager

import torch
import torch.nn as nn
import pytorch_lightning as ptl

import smdistributed.dataparallel.torch.torch_smddp
from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning.plugins.environments import LightningEnvironment

env = LightningEnvironment()
env.world_size = lambda: int(os.environ["WORLD_SIZE"])
env.global_rank = lambda: int(os.environ["RANK"])

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "smddp"

FREEZE_ENCODER = True


def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)
            

def main():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)  ## 100
    parser.add_argument("--model_name", type=str, default='stt_en_quartznet15x5')
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_DATADIR"])
    parser.add_argument("--manifests_dir", type=str, default=os.environ["SM_CHANNEL_MANIFESTS"])

    args, _ = parser.parse_known_args()
    
    
    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    with open(args.data_dir + '/commonvoice_ko.pkl', 'rb') as f:
        train_dev_set = pickle.load(f)
    
    char_model = nemo_asr.models.ASRModel.from_pretrained(args.model_name, map_location='cpu')
    char_model.change_vocabulary(new_vocabulary=train_dev_set)

    
    
    if FREEZE_ENCODER:
        char_model.encoder.freeze()
        char_model.encoder.apply(enable_bn_se)
        logging.info("Model encoder has been frozen, and batch normalization has been unfrozen")
    else:
        char_model.encoder.unfreeze()
        logging.info("Model encoder has been un-frozen")
    
    char_model.cfg.labels = train_dev_set
    
    cfg = copy.deepcopy(char_model.cfg)
    
    train_manifest_cleaned = args.manifests_dir + '/commonvoice_train_manifest_processed.json'
    dev_manifest_cleaned = args.manifests_dir + '/commonvoice_dev_manifest_processed.json'
    test_manifest_cleaned = args.manifests_dir + '/commonvoice_test_manifest_processed.json'

    
    # Setup train, validation, test configs
    with open_dict(cfg):    
        # Train dataset  (Concatenate train manifest cleaned and dev manifest cleaned)
        cfg.train_ds.manifest_filepath = f"{train_manifest_cleaned},{dev_manifest_cleaned}"
        cfg.train_ds.labels = train_dev_set
        cfg.train_ds.normalize_transcripts = False
        cfg.train_ds.batch_size = args.train_batch_size
        cfg.train_ds.num_workers = 8
        cfg.train_ds.pin_memory = True
        cfg.train_ds.trim_silence = True

        # Validation dataset  (Use test dataset as validation, since we train using train + dev)
        cfg.validation_ds.manifest_filepath = test_manifest_cleaned
        cfg.validation_ds.labels = train_dev_set
        cfg.validation_ds.normalize_transcripts = False
        cfg.validation_ds.batch_size = args.test_batch_size
        cfg.validation_ds.num_workers = 8
        cfg.validation_ds.pin_memory = True
        cfg.validation_ds.trim_silence = True
        
    # setup data loaders with new configs
    char_model.setup_training_data(cfg.train_ds)
    char_model.setup_multiple_validation_data(cfg.validation_ds)
    
    
    # Original optimizer + scheduler
    logger.info(OmegaConf.to_yaml(char_model.cfg.optim))
    
    with open_dict(char_model.cfg.optim):
        char_model.cfg.optim.lr = args.lr
        char_model.cfg.optim.betas = [0.95, 0.5]  # from paper
        char_model.cfg.optim.weight_decay = 0.001  # Original weight decay
        char_model.cfg.optim.sched.warmup_steps = None  # Remove default number of steps of warmup
        char_model.cfg.optim.sched.warmup_ratio = 0.05  # 5 % warmup
        char_model.cfg.optim.sched.min_lr = 1e-5
    
    ## Setting Augmentation
    char_model.spec_augmentation = char_model.from_config_dict(char_model.cfg.spec_augment)
    
    ## Setting Metrics
    char_model._wer.use_cer = True
    char_model._wer.log_prediction = True
    
    ddp = DDPPlugin(parallel_devices=[torch.device("cuda", d) for d in range(num_gpus)], cluster_environment=env)
    

    trainer = ptl.Trainer(devices=1, 
                          strategy=ddp, 
                          max_epochs=args.epochs, 
                          accumulate_grad_batches=1,
                          enable_checkpointing=False,
                          logger=False,
                          log_every_n_steps=5,
                          check_val_every_n_epoch=10)

    # Setup model with the trainer
    char_model.set_trainer(trainer)

    # Finally, update the model's internal config
    char_model.cfg = char_model._cfg
    
    
    # Environment variable generally used for multi-node multi-gpu training.
    # In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
    os.environ.pop('NEMO_EXPM_VERSION', None)

    config = exp_manager.ExpManagerConfig(
        exp_dir=f'experiments/lang-{LANGUAGE}/',
        name=f"ASR-Char-Model-Language-{LANGUAGE}",
        checkpoint_callback_params=exp_manager.CallbackParams(
            monitor="val_wer",
            mode="min",
            always_save_nemo=True,
            save_best_model=True,
        ),
    )

    config = OmegaConf.structured(config)

    logdir = exp_manager.exp_manager(trainer, config)
    
    trainer.fit(char_model)

if __name__ == "__main__":
    main()
