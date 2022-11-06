# Initializing all the encoder libraries
import display as display
from IPython.display import Audio
from IPython.utils import io
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import torch

SEED = 10000000

if __name__ == '__main__':
    encoder_weights = Path("saved_models/default/encoder.pt")
    vocoder_weights = Path("saved_models/default/vocoder.pt")
    syn_dir = Path("saved_models/default/synthesizer.pt")
    encoder.load_model(encoder_weights)
    synthesizer = Synthesizer(syn_dir)
    vocoder.load_model(vocoder_weights)

    text = 'Hello all. This is Donald Trump, and I wanted to say that the OneDay is the greatest app in the america.'
    in_fpath = Path("voices/trump10.wav")
    preprocessed_wav = encoder.preprocess_wav(in_fpath)
    # original_wav, sampling_rate = librosa.load(str(in_fpath))
    # preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    embed = encoder.embed_utterance(preprocessed_wav)
    if SEED is not None:
        torch.manual_seed(SEED)
        synthesizer = Synthesizer(syn_dir)
    # with io.capture_output() as captured:
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    if SEED is not None:
        torch.manual_seed(SEED)
        vocoder.load_model(vocoder_weights)
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav = encoder.preprocess_wav(generated_wav)
    display(Audio(generated_wav, rate=synthesizer.sample_rate))