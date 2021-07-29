# ssl-chewing 

A framework for self-supervised feature learning for chewing and eating
detection from in-ear microphone.


## Environment setup

The code uses Keras and TensorFlow v2. See requirements.txt for all
prerequisites, and you can also install them using the following command:

`pip install -r requirements.txt`

An IntelliJ IDEA project is also included and can be optionally used. We suggest
naming the environment as `ssl-chewing` as it the name that is used by the
project.

## Use

### Parameter exploration

To train a self-supervised feature extraction network, and then a chewing
detection network in a LOSO fashion, you can run the following from the source
root directory:

```
python experiments.py \
    --temperature=1.0 \
    --projection_head=linear \
    --g_keep=0 \
    --ta_epochs=100 \
    --ts_epochs=100 \
    --keras_verbosity=2
```

#### Arguments

`--temperature`: Temperature value for self-supervised training.

`--projection_head`: Type of projection head (can be linear or non-linear).

`--g_keep`: Number of layers from the projection head that will be retained as
part of the feature extraction network.

`--ta_epochs`: Number of epochs for self-supervised training.

`--ts_epochs`: Number of epochs for supervised training.

`--keras_verbosity`: Keras verbosity during training and prediction.

### Evaluation

To evaluate a trained feature-extraction network by first training it on a
training set and then evaluating it on a held-out test set, you can run the
following from the source root directory:

```
python experimentspy \
    --part3 \
    --projection_head=non-linear \
    --g_keep=1 \
    --ts_epochs=100 \
    --load_ta_model='<<model-name>>' \
    --keras_verbosity=2 
```

#### Arguments

`--part3`: Flag that selects this evaluation mode.

`--load_ta_model`: [Optional] If provided, a pre-trained feature-extraction
network is loaded from the file.

If you want to retrain the feature extraction network (from inititial random
weights) you can omit the `load_ta_model` argument.


## Cite

IEEE Xplore link: TODO

```
@INPROCEEDINGS{papapanagiotou2021selfsupervised,
  author={Papapanagiotou, Vasileios and Diou, Christos and Delopoulos, Anastasios},
  booktitle={2021 43nd Annual International Conference of the IEEE Engineering in Medicine Biology Society (EMBC)}, 
  title={Self-Supervised Feature Learning of 1D Convolutional Neural Networks with Contrastive Loss for Eating Detection Using an In-Ear Microphone},
  year={2021},
  volume={},
  number={},
  pages={TODO},
  doi={TODO}
}
```

## Notes

This project uses code from the SimCLR framework, available here:
https://github.com/google-research/simclr/ . In particular, it uses the
implementation of LARS optimizer (can be found in
`src/optimizer/larsoptimizer.py`) and the implementation of constrastive loss
(can be found in `src/simclr/contrastiveloss.py`).
