from whisper import log_mel_spectrogram
import matplotlib.pyplot as plt
import numpy as np
import torch
import librosa
from pathlib import Path


def overlay(noise, raw_audio):
    if noise.dim() == 1:
        noise = noise.unsqueeze(0)
    if raw_audio.dim() == 1:
        raw_audio = raw_audio.unsqueeze(0)
    padding = torch.zeros_like(raw_audio[:, :-noise.shape[-1]])
    noise_padded = torch.cat([noise, padding], dim=-1)
    return noise_padded + raw_audio

def audio_to_img(noise, #Learned noise
                 audio_list, #Sample noises to overlay over audio
                 raw_sample, #Raw audio sample to overlay audio over
                 save_dir: Path, #Directory for saving resultant images
                 sampling_rate=16000):
    """
    Takes in audio
    Returns list of paths to images
    """
    paths = []
    audio_list.append(noise)

    for i,audio in enumerate(audio_list):

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        audio = audio.astype(np.float32)

        audio = overlay(audio,raw_sample)
        hop_length = 160
        mel_spec1 = log_mel_spectrogram(audio).numpy().squeeze()

        plt.figure(figsize=(16, 5))  
        librosa.display.specshow(mel_spec1, sr=sampling_rate, hop_length=hop_length, 
                                x_axis='time', y_axis='mel', shading="gouraud", cmap="magma")
        plt.title(f"Unperturbed Audio ({audio.shape[1]/sampling_rate:.2f} seconds)")
        plt.colorbar()
        image_path = save_dir / f"{i}.png"
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        paths.append(save_dir / f"{i}.png")
    # plt.colorbar(format="%+2.0f dB")
    return paths 

from pptx import Presentation
from pptx.util import Inches, Pt

def generate_example_ppt(image_paths, output_path="stacked_images.pptx"):
    if not isinstance(image_paths[0],str):
        image_paths = [str(img) for img in image_paths]

    # Create a PowerPoint presentation
    prs = Presentation()

    # Set slide dimensions (optional, default is widescreen 16:9)
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # Add a blank slide
    slide = prs.slides.add_slide(prs.slide_layouts[5])

    left_space = Inches(6)  # Leave space on the left for audio
    top_margin = Inches(0.5)  # Top margin
    spacing = Inches(0.5)  # Space between images
    img_height = (prs.slide_height - top_margin * 2 - spacing * 4) / 5  # Dynamic height for 5 images

    # Add images stacked vertically
    for i, img_path in enumerate(image_paths):

        top = top_margin + i * (img_height + spacing)
        slide.shapes.add_picture(img_path, left_space, top, height=img_height) # Add images uniformly spaced vertically

        slide.shapes.add_movie(                                                # Add audio at same positions
            r"/home/jaydenfassett/audioversarial/imperceptible/audio_with_attack.wav",
            left=Inches(4.17),
            top=top,       
            width=Inches(1.67),
            height=Inches(0.76),
            poster_frame_image=None,
            mime_type='audio/mp3'
        )
                # Save the presentation
    prs.save(output_path)
    print(f"Presentation saved as {output_path}")


if __name__ == "__main__":
    print("")