from mlflow.tracking import MlflowClient
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def get_callback(save_dir):
    #checkpoint & early stop
    checkpoint_score_callback = ModelCheckpoint(
        dirpath = save_dir,
        filename = 'best_val_acc',
        verbose = True,
        save_last = True,
        save_top_k=1,
        monitor='val_acc',
        mode='max'
    )
    
    checkpoint_loss_callback = ModelCheckpoint(
        dirpath = save_dir,
        filename = 'best_val_loss',
        verbose = True,
        save_last = True,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10, # epochs
        verbose=True,
        mode='min'
    )
    
    return [checkpoint_score_callback, checkpoint_loss_callback, early_stopping]

def print_auto_logged_info(r):
    
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))