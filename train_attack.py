import torch
import pytorch_lightning
from tqdm import tqdm
from pytorch_lightning import Trainer

from src.data import AudioDataModule
from src.attacker import MelBasedAttackerLightning

data_module = AudioDataModule(dataset_name="librispeech",num_workers=0)
model = MelBasedAttackerLightning(sec=1,prepend=True)

trainer = Trainer(max_epochs=5)
trainer.fit(model,data_module)