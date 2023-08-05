import pytorch_lightning as pl
from torch.optim import AdamW
from darts.models import RNNModel, DLinearModel, BlockRNNModel, NLinearModel, TransformerModel, TCNModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

EPOCH = 100
BATCH_SIZE = 64

METHOD_DICT = {
    'EEMD': ['ORIG', 'TREND', 'Imf_0', 'Imf_1', 'Imf_2', 'Imf_3', 'Imf_4', 'Imf_5', 'Imf_6'],
    'VMD': ['ORIG', 'TREND', 'Imf_0', 'Imf_1', 'Imf_2', 'Imf_3', 'Imf_4', 'Imf_5', 'Imf_6', 'Imf_7', 'Imf_8']
}

OPTIM_CLS = AdamW
OPTIM_KWARGS = {'lr': 1e-3, 'weight_decay': 1e-4}


def get_model(name):
    model_dict = {
        'LSTM': BlockRNNModel(input_chunk_length=12, output_chunk_length=1, hidden_dim=64,
                              model='LSTM', model_name='LSTM', batch_size=BATCH_SIZE,
                              show_warnings=True, optimizer_cls=OPTIM_CLS, optimizer_kwargs=OPTIM_KWARGS),
        'TCN': TCNModel(input_chunk_length=12, output_chunk_length=1, model_name='TCN', batch_size=BATCH_SIZE,
                        show_warnings=True, optimizer_cls=OPTIM_CLS, optimizer_kwargs=OPTIM_KWARGS),
        'RNN': BlockRNNModel(input_chunk_length=12, output_chunk_length=1, hidden_dim=64,
                             model='RNN', model_name='RNN', batch_size=BATCH_SIZE,
                             show_warnings=True, optimizer_cls=OPTIM_CLS, optimizer_kwargs=OPTIM_KWARGS),
    }
    return model_dict[name]


def get_trainer(checkpoint_path, log_path):
    return pl.Trainer(enable_checkpointing=True,
                      logger=TensorBoardLogger(log_path, name='', version=''),
                      callbacks=[ModelCheckpoint(dirpath=checkpoint_path),
                                 EarlyStopping(patience=20, monitor='val_loss')])
