"""
Modified decoder file from original whisper repo.
"""
from typing import TYPE_CHECKING

import torch
import whisper

if TYPE_CHECKING:
    pass




class RawDecoder():

    """
    Modified Whisper decoder which pulls out target given sequence
    NOTE: Doesn't support beamsearch
    """
    
    def __init__(self,model,tokenizer,device):
        super(RawDecoder, self).__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.sot_tokens = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=0).to(self.device)
    def update_device(self,device):
        self.model = self.model.to(device)

    def forward(self,mel,tokens=None):
        if type(tokens) is None:
            tokens = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=1).to(self.device)
            
        return self.model.forward(mel,tokens)
    
    def get_eot_prob(self,mel,no_speech: bool = False):
        """
        Runs inference pass on Whisper and returns (<|endoftranscript|>, <|nospeech|> probabilities)
        """
        eot_id = self.tokenizer.eot
        no_speech_id = self.tokenizer.no_speech
        sot_seq = self.tokenizer.sot_sequence_including_notimestamps

        sot_tokens = torch.tensor(sot_seq).unsqueeze(dim=0).to(self.device)
        logits = self.forward(mel,sot_tokens)
        probs = logits.softmax(dim=-1)

        eot_probs = probs[:,-1,eot_id] # Pull out LAST decoded token. This is a sequence of all input tokens and their next values.
        no_speech_probs = probs[:,-1,no_speech_id]
        return eot_probs,no_speech_probs, probs[:,-1,:]

    def transcribe(self,x): # Whisper transcribe
        return self.model.transcribe(x)

    def autoregressive(self,mel,n=30,sot_tokens=None):
        batch_size = mel.shape[0]
        if sot_tokens is None:
            sot_tokens = torch.tensor(self.tokenizer.sot_sequence_including_notimestamps).unsqueeze(dim=0).to(self.device)
        tokens = torch.zeros(size=(batch_size, n + sot_tokens.shape[-1]),dtype=torch.int64).to(self.device)
        
        tokens[:,:sot_tokens.shape[1]] = sot_tokens
        eot_id = self.tokenizer.eot
        for i in range(n):
            # print(i)
            logits = self.forward(mel,tokens[:,:i+sot_tokens.shape[1]])
            # print(logits.a)
            pred = logits[:,-1].argmax(dim=-1)
            tokens[0:,i+sot_tokens.shape[1]] = pred
            # print(pred)

            printable = tokens.squeeze()
            printable = printable[printable != 0]
            # if i % 2 != 0:
            print(self.tokenizer.decode(printable))
            # print(pred)
            # print(eot_id)
            # exit()
            if pred.item() == eot_id:
                # print("break")
                break
        return tokens

if __name__ == "__main__":
    unperturbed = whisper.load_audio("/home/jaydenfassett/audioversarial/imperceptible/demo/sample.wav")
    perturbed = whisper.load_audio("/home/jaydenfassett/audioversarial/imperceptible/demo/sample_attacked.wav")
    device = "cuda:3"
    # print("before:",x.shape)
    # x = np.concatenate([x,x,x])
    # print("after",x.shape)
    def prep(x):
        x = whisper.pad_or_trim(x)
        mel = whisper.log_mel_spectrogram(x).to(device).unsqueeze(dim=0)
        return mel

    # mels = torch.cat((prep(q),prep(x)),dim=0)
    

    # mels = torch.cat((prep(q),prep(x)),dim=0).to(device)
    # print(mels.shape)

    NAME = "tiny.en"
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
    model = whisper.load_model(NAME)
    # mel = mels[1].unsqueeze(0)
    mel = prep(perturbed)

    qq = RawDecoder(model=model,tokenizer=tokenizer,device=device)

    qq.autoregressive(mel,n=50,sot_tokens=qq.sot_tokens)

    # print(qq.get_eot_prob(mel))