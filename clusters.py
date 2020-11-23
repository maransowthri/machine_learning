import os
import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.3]])
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution, 
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7
    )

mean = model.mean()
os.system('CLS')
print(mean)

with tf.compat.v1.Session():
    print(mean.numpy())
