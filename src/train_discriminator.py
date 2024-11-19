import torch
import torchaudio
import torch.nn as nn
from tqdm import tqdm
from typing import Union
from pathlib import Path
from whisper import log_mel_spectrogram, pad_or_trim
from .discriminator import MelDiscriminator
from .data import AudioDataModule

#TODO: Add raw audio discriminator




def train_mel(discriminator,
              save_dir:Union[str, Path],
              attack_length_sec: float = 1.,
              dataset:str = "librispeech",
              device: str = "cuda",
              n_epochs: int = 15,
              batch_size: int = 128, #Only half will be real audio, other half will be random noise
              optimizer: str = "adam",
              num_workers = 4  
              ):
    
    half_batch = batch_size // 2
    dataloader = AudioDataModule(dataset,
                              batch_size=half_batch,
                              num_workers=num_workers).train_dataloader()
    

    loss_fn = nn.BCELoss()
    if optimizer == "adam":
        optimizer = torch.optim.Adam(discriminator.parameters())


    attack_n = int(attack_length_sec*100)
    pbar = tqdm(range(n_epochs))
    for epoch in pbar:

        losses = []
        running_loss = 0
        for batch in dataloader:
            raw_audio, sampling_rate, transcript = batch
            real = log_mel_spectrogram(pad_or_trim(raw_audio))[:,:,:attack_n].to(device)
            real = real.unsqueeze(1) #Adding channel dimension
            noise = torch.randn((half_batch,1,80,attack_n)).to(device) # (Half_batch, 1, 80 ,LENGTH)

            labels = torch.zeros(batch_size).to(device)
            labels[half_batch:] = 1 # (0,0,0.... 1,1,1)

            # print(f"Noise shape: {noise.shape} \n X Shape: {real.shape}")

            shuffle_index = torch.randperm(batch_size).to(device) # Shuffle order

            x = torch.cat([real,noise],dim=0) #Concatenate along batch_size dimension
            x = x[shuffle_index]
            labels = labels[shuffle_index] #Shuffle input & labels


            optimizer.zero_grad()

            y_pred = discriminator(x).squeeze(dim=-1)


            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        losses.append(loss.item())
        pbar.set_description(f"Loss = {avg_loss:.4f}")
    output_dir = save_dir / f"discriminator_{attack_n}tsteps.pth"
    print(f"Saving at {output_dir}")

    torch.save(discriminator.state_dict(), output_dir)
        # tqdm.write(f"Loss = {avg_loss:.4f}")
    return

if __name__ == "__main__":
    MODEL_DIR = Path(__file__).parent.parent / "discriminator"
    D = MelDiscriminator().to("cuda")
    train_mel(D,MODEL_DIR,n_epochs=3,device="cuda")
    # weights = torch.load(MODEL_DIR / "discriminator_weights.pth",weights_only=True)
    # D.load_state_dict(weights)



    aud = AudioDataModule().val_dataloader()

    samp = next(iter(aud))
    samp = log_mel_spectrogram(pad_or_trim(samp[0]))[:,:,:100].unsqueeze(1).to("cuda")
    print(samp.shape)
    print(D(samp).mean())
