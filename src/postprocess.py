import pandas as pd
import numpy as np
from src.visual_utils import save_photo_overlay
from src.data import AudioDataModule
from src.utils import overlay_np
from pathlib import Path
from tqdm import tqdm
from scipy.io.wavfile import write
import argparse


def normalize_audio(audio):
    audio = audio.astype(np.float32)
    audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]
    return (audio * 32767).astype(np.int16)  # Convert to int16 for WAV
def make_report(df, test_name, dataset, dir_name, metric_name=None):
    if not isinstance(df,pd.DataFrame):
        df = pd.read_csv(df)
    df['img_paths'] = df['noise_path'].str.replace("/noise/", "/examples/images/")
    df['img_paths'] = df['img_paths'].str.replace("/noise.np.npy", "/plot.png")
    dm = AudioDataModule(dataset)
    sample = dm.sample[0].squeeze().numpy()
    sampling_rate = dm.sample[1]
    test = df[df['Name'] == test_name]

    if metric_name is not None:
        test = test.sort_values(by=metric_name)
    else:
        test = test.sort_index()
    save_dir = Path(dir_name) / test_name
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save unperturbed audio
    write(save_dir / "unperturbed.wav", sampling_rate, sample)

    # Start HTML content
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Test Report: {test_name}</title>
</head>
<body>
  <h1>Test Report: {test_name}</h1>
  <h2>Original Audio</h2>
  <audio controls>
    <source src="unperturbed.wav" type="audio/wav">
    Your browser does not support the audio element.
  </audio>
"""

    for i, row in tqdm(test.iterrows(), total=len(test)):
        gamma = row.gamma
        val_per = row.val_per
        clip_val = row.clip_val
        per_muted = row.per_muted
        if metric_name is not None:
            targ = row[metric_name]
            targ = round(float(targ),3)
        else:
            targ = i
        # break

        # if not isinstance(clip_val,float):
        #     clip_val = clip_val[1]
        # print(tuple(targ))
        
        # exit()
        # if isinstance(targ):

        noise = np.load(row['noise_path']).squeeze()
        wavefile = normalize_audio(overlay_np(noise, sample))

        out_wav = f"{i}.wav"
        out_img = f"{i}.png"

        write(save_dir / out_wav, sampling_rate, wavefile)
        save_photo_overlay(noise, sample, save_dir / out_img)

        html += f"""
  <hr>
  <h2>{metric_name}: {targ} | % muted: {per_muted:.2f}  | SNR {row.snr:.2f}</h2>
  <audio controls>
    <source src="{out_wav}" type="audio/wav">
    Your browser does not support the audio element.
  </audio>
  <br>
  <img src="{out_img}" alt="Overlay Plot for {metric_name} {targ}" style="max-width:600px;"><br>
    <small>{row.noise_path}</small>

"""

    html += """
</body>
</html>
"""

    # Write final HTML
    report_path = save_dir / "report.html"
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"HTML report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_value", help="Experiment path")
    parser.add_argument("metric_name",help="Which metric to track")
    args = parser.parse_args()
    make_report("/home/jaydenfassett/audioversarial/imperceptible/paths.csv",args.input_value,"tedlium:","/home/jaydenfassett/audioversarial/imperceptible/exampletest",args.metric_name)


