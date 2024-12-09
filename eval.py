import argparse
import whisper
import torch
import numpy as np
from jiwer import wer
import torchaudio
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pathlib import Path
from whisper import pad_or_trim
from tqdm import tqdm


# Set the data directory
root_dir = Path(__file__).parent.parent
data_dir = Path("/media/drive1/jaydenfassett/audio_data")


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name="librispeech", batch_size=32, num_workers=0):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.setup()
    def prepare_data(self):
        # Download the dataset if it is not already available TODO: Full conditionals
        # if self.dataset_name == "librispeech":
        self.dataset = torchaudio.datasets.LIBRISPEECH(data_dir,
                            # 'dev-clean',
                            # 'train-clean-100',
                            self.dataset_name,
                            # 'train-other-500',
                                    download = True)
    
    def setup(self, stage=None):
        self.prepare_data()
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def collate_fn(self, batch):
        # Collate function to handle variable-length audio sequences

        audio, sampling_rate,transcript,utt,speak,chap= zip(*batch)

        #No need to trim 
        resized_audio = [pad_or_trim(aud) for aud in audio]
        audio_signals = torch.cat(resized_audio,dim=0)
        # labels = torch.tensor(labels)
        return audio_signals, sampling_rate, transcript

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)



def overlay(noise, raw_audio):
    # Ensure noise and raw_audio are on the same device
    noise = noise.to(raw_audio.device)

    # Pad noise to match the length of raw_audio
    if noise.shape[-1] > raw_audio.shape[-1]:
        noise = noise[:, :raw_audio.shape[-1]]
    else:
        pad_length = raw_audio.shape[-1] - noise.shape[-1]
        noise = torch.nn.functional.pad(noise, (0, pad_length))

    return noise + raw_audio


def get_args():
    parser = argparse.ArgumentParser(description="Training Script Arguments")

    # General training settings
    parser.add_argument('--model', type=str, default='tiny.en', help='Which Whisper model')
    parser.add_argument('--metric', choices=['wer'], default='wer', help='Which metric to use')
    parser.add_argument('--noise_path', type=str, required=True, help='Path to noise vector')
    parser.add_argument('--dataset', type=str, choices=['dev-clean', 'train-clean-100', 'train-other-500'], required=True)
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    args = parser.parse_args()
    return args


def main(args):
    # Load the Whisper model and move it to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(args.model).to(device)

    # Load the dataset
    data_module = AudioDataModule(args.dataset, args.batch_size, num_workers=args.num_workers)
    data_module.prepare_data()
    data_loader = data_module.train_dataloader()

    # Load the noise vector as a tensor and move it to GPU
    noise = torch.tensor(np.load(args.noise_path), dtype=torch.float32).to(device)

    # Initialize lists to store references and predictions
    references = []
    predictions = []

    # Loop through the dataset
    num_processed = 0
    total_wer = 0.0
    pbar = tqdm(data_loader, desc="Processing Batches")
    for audio_signals, sr,transcripts in pbar:

        audio_signals = audio_signals.to(device)  # Move audio signals to GPU

        # Apply adversarial noise using the overlay function
        perturbed_audio = overlay(noise, audio_signals)

        # Loop through each perturbed audio sample in the batch
        for audio, ref_transcript in zip(perturbed_audio, transcripts):
            result = model.transcribe(audio, language='en')
            predicted_text = result['text']

            # Store references and predictions
            references.append(ref_transcript)
            predictions.append(predicted_text)

            # Update running WER
            current_wer = wer([ref_transcript], [predicted_text])
            total_wer += current_wer
            num_processed += 1

            # Update the progress bar description with the running WER
            # tqdm.write(f"Running WER: {total_wer / num_processed:.4f}")
            pbar.set_description(f"Running WER: {total_wer / num_processed:.4f}")
    computed_wer = wer(references, predictions)
    # Print the WER result
    print(f"Word Error Rate (WER) over the dataset: {computed_wer:.4f}")



if __name__ == "__main__":
    args = get_args()
    main(args)
