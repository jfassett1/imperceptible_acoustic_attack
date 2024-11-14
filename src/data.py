import torchaudio
from pathlib import Path

root_dir = Path(__file__).parent.parent
data_dir = root_dir / "data"


# print(root_dir)
librispeech = torchaudio.datasets.LIBRISPEECH(data_dir,
                                 'dev-clean',
                                   'LibriSpeech',
                                     download = True)


iterator = iter(librispeech)
sample = next(iterator)
print(sample)
# print(librispeech)