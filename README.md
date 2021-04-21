# cnn-cifar10 <a href="https://web.spell.ml/workspace_create?workspaceName=cnn-cifar10&githubUrl=https%3A%2F%2Fgithub.com%2Fspellml%2Fccn-cifar10&pip=pillow"><img src=https://spell.ml/badge.svg height=20px/></a>

A simple PyTorch CNN trained on the CIFAR-10 dataset. Comes in a few different flavors:

* `models/train_basic.py`&mdash;simple no-frills training script.
* `models/train.py`&mdash;training script with optional parameters, useful for e.g. [hyperparameter search](https://spell.ml/docs/hyper_searches/). Also incorporates [TensorBoard support](https://spell.ml/docs/tensorboard_tutorial/).
* `models/distributed_train.py`&mdash;training script with [Horovod](https://github.com/horovod/horovod) support, to demonstrate Spell's [distributed training](https://spell.ml/docs/distributed_runs/) feature.
* `models/wandb_train.py`&mdash;training script leveraging Spell's [Weights & Biases integration](https://spell.ml/docs/integrating_wandb/).
* `servers/serve.py`&mdash;example [model server script](https://spell.ml/docs/model_servers/) compatible with any of the models in this repo.

Refer to the `notebooks/` folder for implementation notes.

See also [the Spell quickstart](https://spell.ml/docs/quickstart/).
