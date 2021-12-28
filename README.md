# MelGAN-Waveform-synthesis
Pytorch re-implementation of [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://papers.nips.cc/paper/2019/hash/6804c9bca0a615bdb9374d00a9fcba59-Abstract.html).


## What is Text To Speech (TTS)?
- Voice synthesis via converting text input into audio output
- Audio modeling
  > Stage 1. Models the intermediate representation given text as input
  
  > Stage 2. Transforms the intermediate representation back to audio (i.e., vocoder)
- Representation
  - Typically chosen to be easier to model than raw audio while preserving enough information to allow faithful inversion back to audio

![fig2](https://user-images.githubusercontent.com/57162425/147518655-6a731bf4-ad05-435f-a3c9-54770fe4a79d.png)

## Basic Knowledge
- Mel-spectrogram (example data from LJ Speech dataset [LJ001-0001])
- LJ Speech dataset:
  > This is consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books.
  
  > Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours.

![fig1](https://user-images.githubusercontent.com/57162425/147518659-9f056950-115e-4a73-8a77-a66831d83097.png)


## MelGAN
- **Generator**
  - The generator is a fully convolutional feed-forward network with mel-spectrogram as input and raw waveform as output.
  - Stack of transposed convolutional layers are used to upsample the input sequence and each trnasposed convolutional layers is followed by a stack of residual blocks with dilated convolutions.

- **Discriminator**
  - Multi-scale architecture with 3 discriminators that have identical network structure but operate on different audio scales are adopted.
  - This structure has an inductive bias that each discriminator learns features for different frequency range of the audio.


![gd](https://user-images.githubusercontent.com/57162425/147518660-b03f88da-82e5-485a-b747-ba5627c74320.png)
