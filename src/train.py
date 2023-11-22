import os
import sys
import argparse
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

sys.path.append('..')
from src.config import *
from src.dataset import DataModule
from src.model import LightningNetwork

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='PyTorch Lightning training arg parser')
    # parser.add_argument(
    #     '--base_model_name', 
    #     type=str,
    #     help='The name of the base model.',
    # )
    # parser.add_argument(
    #     '--dropout', 
    #     type=float,
    #     help='The dropout rate to be used.',
    # )
    # parser.add_argument(
    #     '--output_dims', 
    #     type=list,
    #     nargs='+',
    #     help='A list containing the number of neuron in each layer',
    # )
    # parser.add_argument(
    #     '--lr', 
    #     type=float,
    #     help='The learning rate to be used.',
    # )
    # parser.add_argument(
    #     '--trail_name', 
    #     type=str,
    #     help='The name of the experiment for tensorboard logging.',
    # )
    # parser.add_argument(
    #     '--task', 
    #     type=str,
    #     help='Either of `train`, `val`, `test` or `predict`.',
    # )
    # args = parser.parse_args()

    # logger = TensorBoardLogger(
    #     save_dir=TB_LOG_DIR, 
    #     name=args.trail_name
    # )
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{TB_LOG_DIR}/{args.trail_name}_profiler'),
    #     schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    # )

    # model = LightningNetwork(
    #         base_model=get_base_model(args.base_model_name),
    #         dropout=args.dropout, 
    #         output_dims=args.output_dims,
    #         learning_rate=args.lr
    # )
    # data_module = DataModule(model_name=args.base_model_name)

    # trainer = pl.Trainer(
    #     profiler=profiler,
    #     logger=logger,
    #     accelerator=ACCELERATOR,
    #     devices=DEVICES,
    #     max_epochs=NUM_EPOCHS,
    #     precision=PRECISION,
    #     fast_dev_run=True
    #     # callbacks=[EarlyStopping(monitor="val_loss")],
    # )
    # if args.task =='train':
    #     trainer.fit(model, data_module)
    # elif args.task == 'val':
    #     trainer.validate(model, data_module)
    # elif args.task == 'test':
    #     trainer.test(model, data_module)
    # else:
    #     trainer.predict(model, data_module)


    base_model_name = 'efficientnet_b0'
    model = LightningNetwork(
            base_model=get_base_model(base_model_name),
            dropout=0.2, 
            output_dims=[128, 64],
            learning_rate=LEARNING_RATE
    )
    data_module = DataModule(model_name=base_model_name)

    trainer = pl.Trainer(
        logger=True,
        max_epochs=NUM_EPOCHS,
        precision=PRECISION,
        # max_steps=5,
        # fast_dev_run=True
        # callbacks=[EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(model, data_module)




# python train.py --base_model_name efficientnet_b0 --dropout 0.5 --output_dims 128 64 32 --lr 0.001 --trail_name experiment1 --task train