from pycomposer.gancomposable._torch.conditional_gan_composer import ConditionalGANComposer
import torch
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
import matplotlib.pyplot as plt
import seaborn as sns

logger = getLogger("pygan")
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
plt.style.use("fivethirtyeight")

gan_composer = ConditionalGANComposer(
    # `list` of Midi files to learn.
    midi_path_list=[
        'audio.mid'
    ], 
    # Batch size.
    batch_size=20,
    # The length of sequence that LSTM networks will observe.
    seq_len=8,
    # Learning rate in `Generator` and `Discriminator`.
    learning_rate=1e-10,
    # Time fraction or time resolution (seconds).
    time_fraction=0.5,
)

gan_composer.learn(
    # The number of training iterations.
    iter_n=1000, 
    # The number of learning of the `discriminator`.
    k_step=10
)

gan_composer.compose(
    # Path to generated MIDI file.
    file_path="generated.mid", 
    # Mean of velocity.
    # This class samples the velocity from a Gaussian distribution of 
    # `velocity_mean` and `velocity_std`.
    # If `None`, the average velocity in MIDI files set to this parameter.
    velocity_mean=None,
    # Standard deviation(SD) of velocity.
    # This class samples the velocity from a Gaussian distribution of 
    # `velocity_mean` and `velocity_std`.
    # If `None`, the SD of velocity in MIDI files set to this parameter.
    velocity_std=None
)

# Plot the loss of `Generator` and `Discriminator`.
sns.lineplot(data=gan_composer.loss_df)
plt.show()
