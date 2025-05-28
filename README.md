# VAE Agent

This Master's thesis project explores the application of Variational Autoencoders (VAEs) to enhance the robustness and adaptability of autonomous driving systems that account for uncertainty in its decision-making process. The proposed approach, VAEAgent, involves training a VAE to model the distribution of a latent vector obtained by passing driving scenarios through an encoder model. Sampling from this distribution generates new representations of the latent vector. Taking an average of these samples allows the agent to work with a more generalised latent vector and make more robust decisions, particularly in uncertain and novel environments.

<p align="center">
  <img width="300" height="300" src="/images/overview_model.PNG">
</p>

## Results
The experimental results demonstrate the effectiveness of the proposed approach when compared with similarly trained models, highlighting the potential of generative models to improve the safety and reliability of autonomous driving systems. The VAEAgent was tested on the CARNOVEL benchmark, where it was evaluated on scenes unseen to it from training. Here are the comparisons of VAEAgent with other uncertainty-aware methods.


<p align="center">
  <b>Deep Imitative Model (DIM)</b>
  <br>
  <img src="/images/DIM_RA_1.gif">
</p>

<p align="center">
  <b>VAEAgent</b>
  <br>
  <img src="/images/VAEAgent_RA_1.gif">
</p>

<p align="center">
  <img width="800" height="200" src="/images/model_results.PNG">
</p>
