# MelGAN-Waveform-synthesis
Pytorch re-implementation of [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://papers.nips.cc/paper/2019/hash/6804c9bca0a615bdb9374d00a9fcba59-Abstract.html).


## What is Text To Speech (TTS)?
- Voice synthesis via converting text input into audio output
- Audio modeling
  > Stage 1. Models the intermediate representation given text as input
  
  > Stage 2. Transforms the intermediate representation back to audio (i.e., vocoder)
- Representation
  > Typically chosen to be easier to model than raw audio while preserving enough information to allow faithful inversion back to audio

## Basic Knowledge
- Mel-spectrogram (example data from LJ Speech dataset [LJ001-0001])

![gd](https://user-images.githubusercontent.com/57162425/147517884-359a3315-c465-4045-8229-d3565692b5d7.png)
