# Unimodulra Regularization
[![arXiv](https://img.shields.io/badge/arXiv-2501.12116-b31b1b.svg)](https://arxiv.org/abs/2501.12116)

We present in this Github the Unimodular regularization techinque already presented in the following article: arXiv:2501.12116. This technique has been implemented for Physics Informed Neural Networs. Particularly,
used on a Multihead setup. It has been proven that this techique allows a better response of the latent space with respect to changes on the inputs. This allows the to achieve better solutions when transfer
learning the solution to a stiffer regime in which the original model has been trainend on.

## How does this work?
This setup is implemented in the alredy build [pytorch](https://github.com/pytorch/pytorch) package [neurodiffeq](https://github.com/NeuroDiffGym/neurodiffeq) to solve differential equations using neural networks. This technique consists on embedding the latent space into the
input space

$$\Omega =  (x^\mu, H_i(x^\mu))$$

where $x^\mu$ represents the input coordinates and $H_i(x^\mu)$ the components of the latent space produced by the body of the multihead. Then, inspired by differential geometry techniques, we can compute the metric
tensor of this space as

$$g_{\mu\nu} = \frac{\partial \Omega}{\partial x^\mu} \frac{\partial \Omega}{\partial x^\nu}$$

At last, we will compute the metric determinant $g = \det(g_{\mu\nu})$ and impose that it must be equal to one. We have put this constraint as an additional loss of the model. Intuitevely, we are forcing the model
to not present high variability of the latent space when the inputs are slightly modified.

## Organization 
The organization of the Github is as follows: on the first folder (main_paper_results) the setup to train and generate the results of the paper (arXiv:2501.12116) can be found. We have included the three cases taken
into account (flame equation, Van der Pol oscillator and holography).

We have also added an additional folder called test, where we have put some test performed that didn't work as well as we thought. For more details, please check the main article of this GitHub: arXiv:2501.12116
