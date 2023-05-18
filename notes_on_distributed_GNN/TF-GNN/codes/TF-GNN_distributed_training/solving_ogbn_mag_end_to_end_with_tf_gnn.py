# -*- coding: utf-8 -*-
"""Solving OGBN-MAG end-to-end with TF-GNN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_e2e.ipynb

##### Copyright 2022 The TensorFlow GNN Authors.

Licensed under the Apache License, Version 2.0 (the "License");
"""

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eicther express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""# Solving OGBN-MAG end-to-end with TF-GNN

<table class="tfo-notebook-buttons" align="left">
  <td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_e2e.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/gnn/blob/main/examples/notebooks/ogbn_mag_e2e.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View on GitHub</a>
  </td>
</table>

### Abstract

[Graph Neural Networks](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/intro.md) (GNNs) are a powerful tool for deep learning on relational data. This tutorial introduces the two main tools required to train GNNs at scale:

1. *Graph Sampler*: The [graph sampler](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/sampler/graph_sampler.py) helps to efficiently sample subgraphs from huge graphs.
2. *The Runner*: Also known as the Orchestrator, [the runner](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/runner.md) orchestrates the end to end training of GNNs with minimal coding. The Runner code is a high-level abstraction for training GNNs models provided by the TensorFlow GNN (TF-GNN) library.

This tutorial is intended for ML practitioners with a basic idea of GNNs.

## Colab set-up
"""

!pip install -q tensorflow-gnn || echo "Ignoring package errors..."

import functools
import os
import re

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
from tensorflow_gnn.models import vanilla_mpnn

print(f"Running TF-GNN {tfgnn.__version__} under TensorFlow {tf.__version__}.")

"""## Introduction

### Problem statement and dataset

OGBN-MAG is [Open Graph Benchmark](https://ogb.stanford.edu)'s Node classification task on a subset of the [Microsoft Academic Graph](https://www.microsoft.com/en-us/research/publication/microsoft-academic-graph-when-experts-are-not-enough/).

The OGBN-MAG dataset is one big heterogeneous graph. The graph has four sets (or types) of nodes.

  * Node set "paper" contains 736,389 published academic papers, each with a 128-dimensional word2vec feature vector computed by averaging the embeddings of the words in its title and abstract.
  * Node set "field_of_study" contains 59,965 fields of study, with no associated features.
  * Node set "author" contains the 1,134,649 distinct authors of the papers, with no associated features.
  * Node set "institution" contains 8740 institutions listed as affiliations of authors, with no associated features.

The graph has four sets (or types) of directed edges, with no associated features on any of them.

  * Edge set "cites" contains 5,416,217 edges from papers to the papers they cite.
  * Edge set "has_topic" contains 7,505,078 edges from papers to their zero or more fields of study.
  * Edge set "writes" contains 7,145,660 edges from authors to the papers that list them as authors.
  * Edge set "affiliated_with" contains 1,043,998 edges from authors to the zero or more institutions that have been listed as their affiliation(s) on any paper.

The task is to **predict the venue** (journal or conference) at which each of the papers has been published. There are 349 distinct venues, not represented in the graph itself. The benchmark metric is the accuracy of the predicted venue.

Results for this benchmark confirm that the graph structure provides a lot of relevant but "latent" information. Baseline models that only use the one explicit input feature (the word2vec embedding of a paper's title and abstract) perform less well.

OGBN-MAG defines a split of node set "papers" into **train, validation and test nodes**, based on its "year" feature:

  * "train" has the 629,571 papers with `year<=2017`,
  *  "validation" has the 64,879 papers with `year==2018`, and
  * "test" has the 41,939 papers with `year==2019`.

However, under OGB rules, training may happen on the full graph, just restricted to predictions on the "train" nodes. We follow that for consistency in benchmarking. However, users working on their own datasets may wish to validate and test with a more realistic separation between training data from the past and evaluation data representative of future inputs for prediction.

### Approach

OGBN-MAG asks to classify each of the "paper" nodes. The number of nodes is on the order of a million, and we intuit that the most informative other nodes are found just a few hops away (cited papers, papers with overlapping authors, etc.).

Therefore, and to stay scalable for even bigger datasets, we approach this task with **graph sampling**: Each "paper" node becomes one training example, expressed by a subgraph that has the node to be classified as its root and stores a sample of its neighborhood in the original graph. The sample is taken by going out a fixed number of steps along specific edge sets, and randomly downsampling the edges in each step if they are too numerous.

The actual **TensorFlow model** runs on batches of these sampled subgraphs, applies a Graph Neural Network to propagate information from related nodes towards the root node of each batch, and then applies a softmax classifier to predict one of 349 classes (each venue is a class).

The exponential fan-out of graph sampling quickly gets expensive. Sampling and model should be designed together to make the most of the available information in carefully sampled subgraphs.

## Data preparation and graph sampling

TF-GNN's [Data Preparation and Sampling](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/data_prep.md) guide explains how to sample subgraps from a big input graph with the [graph_sampler](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/sampler/graph_sampler.py) tool, and what its expected input format is. (TF-GNN comes with a [converter script](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/converters/ogb/convert_ogb_dataset.py) from the original OGB format.)

The sampling expected by the model in this colab proceeds as follows:

  1. Start from all "paper" nodes.
  2. For each paper from 1, follow a random sample of "cites" edges to other "paper" nodes.
  3. For each paper from 1 or 2, follow a random sample of reversed "writes" edges to "author" nodes and store them as edge set "written".
  4. For each author from 3, follow a random sample of "writes" edges to more "paper" nodes.
  5. For each author from 3, follow a random sample of "affiliated_with" edges to "institution" nodes.
  6. For each paper from 1, 2 or 4, follow a random sample of "has_topic" edges to "field_of_study" nodes.

The sampling output contains all nodes and edges traversed by sampling, in their respective node/edge sets and with their associated features. An edge between two sampled nodes that exists in the input graph but has not been traversed by sampling is not included in the sampled output. For example, we get the "cites" edges followed in step 2, but no edges for citations between the papers discovered in step 2. Moreover, while edge set "written" is defined in the sampler input as reversal of edge set "writes", the sampler output has different edges in these edge sets (namely those traversed in steps 2 and 3, resp.).

## Reading the sampled subgraphs

The result of sampling is available here (subject to this [license](https://storage.googleapis.com/download.tensorflow.org/data/ogbn-mag/sampled/v1/edge/LICENSE.txt)):
"""

input_file_pattern = "gs://download.tensorflow.org/data/ogbn-mag/sampled/v1/edge/samples-?????-of-00100"
graph_schema_file = "gs://download.tensorflow.org/data/ogbn-mag/sampled/v1/edge/schema.pbtxt"
graph_schema = tfgnn.read_schema(graph_schema_file)
example_input_graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

"""Training a neural network with stochastic gradient descent requires randomly shuffled training data, but ours is too big to fully reshuffle it on the fly while reading. The sampler has assigned a quasi-random key to each subgraph (see the `seed_op` for details) which are then saved as sharded outputs. For speed, we want to read from several shards in parallel.

For distributed training (introduced in a bit), each trainer replica reads from its own subset of the shards. To achieve some randomization between training runs, each replica reshuffles the order of input shards and then shuffles examples within a moderate window.

## Data Split Preparation

The following helper lets us filter a single input dataset by OGBN-MAG's specific rule for the test/validation/train split before parsing the full GraphTensor. (Models for production systems should probably use separate datasets.)
"""

def _is_in_split(split_name: str):
  """Returns a `filter_fn` for OGBN-MAG's dataset splitting."""
  def filter_fn(serialized_example):
    features = {
        "years": tf.io.RaggedFeature(tf.int64, value_key="nodes/paper.year")
    }
    years = tf.io.parse_single_example(serialized_example, features)["years"]
    year = years[0]  # By convention, the root node is the first node
    if split_name == "train":
      return year <= 2017  # 629,571 examples
    elif split_name == "validation":
      return year == 2018  # 64,879 examples
    elif split_name == "test":
      return year == 2019  # 41,939 examples
    else:
      raise ValueError(f"Unknown split_name: '{split_name}'")
  return filter_fn

class _SplitDatasetProvider:
  """Splits a `delegate` for OGBN-MAG.

  The OGBN-MAG datasets splits test/validation/train by paper year. This class
  filters a `delegate` with the entire OGBN-MAG dataset by the split name
  (test/validation/train).
  """

  def __init__(self, delegate: runner.DatasetProvider, split_name: str):
    self._delegate = delegate
    self._split_name = split_name

  def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
    dataset = self._delegate.get_dataset(context)
    return dataset.filter(_is_in_split(self._split_name))

# NOTE: For your own file system or GCS bucket, you only need to input the
# `file_pattern` to TFRecordDatasetProvider. For gs://download.tensorflow.org,
# we avoid listing the filenames manually and provide the `_glob_sharded()`
# helper.
def _glob_sharded(file_pattern):
  match = re.fullmatch(r"(.*)-\?\?\?\?\?-of-(\d\d\d\d\d)", file_pattern)
  if match is None:  # No shard suffix found.
    return [file_pattern]
  basename = match[1]
  n = int(match[2])
  return [f"{basename}-{i:05d}-of-{n:05d}" for i in range(n)]

# Split numbers obtained from _is_in_split()
num_training_examples = 629571
num_validation_examples = 64879
num_test_examples = 41939

# The DatasetProvider provides a tf.data.Dataset.
filenames = _glob_sharded(input_file_pattern)
ds_provider = runner.TFRecordDatasetProvider(filenames=filenames)
train_ds_provider = _SplitDatasetProvider(ds_provider, "train")
valid_ds_provider = _SplitDatasetProvider(ds_provider, "validation")

"""## Distributed Training



We use TensorFlow's [Distribution Strategy](https://www.tensorflow.org/guide/distributed_training) API to write a model that can run on multiple TPUs, multiple GPUs, or maybe just locally on CPU.


"""

try:
  tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
  print("Running on TPU ", tpu_resolver.cluster_spec().as_dict()["worker"])
except:
  tpu_resolver = None

if tpu_resolver:
  print("Using TPUStrategy")
  min_nodes_per_component = {"paper": 1}
  strategy = runner.TPUStrategy()
  train_padding = runner.FitOrSkipPadding(example_input_graph_spec, train_ds_provider, min_nodes_per_component)
  valid_padding = runner.TightPadding(example_input_graph_spec, valid_ds_provider, min_nodes_per_component)
elif tf.config.list_physical_devices("GPU"):
  print(f"Using MirroredStrategy for GPUs")
  gpu_list = !nvidia-smi -L
  print("\n".join(gpu_list))
  strategy = tf.distribute.MirroredStrategy()
  train_padding = None
  valid_padding = None
else:
  print(f"Using default strategy")
  strategy = tf.distribute.get_strategy()
  train_padding = None
  valid_padding = None
print(f"Found {strategy.num_replicas_in_sync} replicas in sync")

"""As you might have noticed above, we need to provide a padding strategy when we want to train on TPUs. Next, we explain the need for paddings on TPU and the different padding strategies employed during training and validation.

### Padding (for TPUs)


Training on TPUs involves just-in-time compilation of a TensorFlow model to TPU code, and requires *fixed shapes* for all Tensors involved. To achieve that for graph data with variable numbers of nodes and edges, we need to pad each input Tensor to some fixed maximum size. For training on GPUs or CPU, this extra step is not necessary.

#### TightPadding: Padding for the validation dataset

For the validation dataset, we need to make sure that every batch of examples fits within the fixed size, no matter how the parallelism in the input pipeline ends up combining examples into batches. Therefore, we use a rather generous estimate, basically scaling each Tensor's observed maximum size by a factor of `batch_size`. If that were to run into limitations of accelerator memory, we'd rather shrink the batch size than lose examples.

The dataset in this example is not too big, so we can scan it within a few minutes to determine constraints large enough for all inputs. (For huge datasets under your control, it may be worth inferring an upper bound from the sampling spec instead.)

#### FitOrSkipPadding: Padding for the training dataset

For the training dataset, TF-GNN allows you to optimize more aggressively for large batch sizes: size constraints satisfied by 100% of the inputs have to accommodate the rare combination of many large examples in one batch.

Instead, we use size constraints that will fit *close to* 100% of the randomly drawn training batches. This is not covered by the theory supporting stochastic gradient descent (which calls for examples drawn independently at random), but in practice, it often works, and allows larger batch sizes within the limits of accelerator memory, and hence faster convergence of the training.

## Model Building and Training

We build a model on sampled subgraphs that predicts one of 349 classes (venues) for the subgraph's root node. We use a Graph Neural Network (GNN) to propagate information along edge sets towards the subgraph's root node.

Observe how the various node sets play different roles:

  * Node set "paper" has many nodes. It contains the node to predict on. Some of its nodes are linked by "cites" edges, which seem relevant for the prediction task. Its nodes also carry the only input feature besides adjacency, namely the word2vec embedding of title and abstract.
  * Node set "author" also has many nodes. Authors have no features of their own, but having an author in common provides a seemingly relevant relation between papers.
  * Node set "field_of_study" has relatively few nodes. They have no features by themselves, but having a common field of study provides a seemingly relevant relation between papers.
  * Node set "institution" has relatively few nodes. It provides an additional relation on authors.

For node sets "paper" and "author", we follow the standard GNN approach to maintain a hidden state for each node and update it several times with information from the inbound edges. Notice how sampling has equipped each "paper" or "author" adjacent to the root node with a 1-hop neighborhood of its own. Our model does 4 rounds of updates, which covers the longest possible path in a sampled subgraph: a seed paper "cites" a paper that was "written" by an author who "writes" another paper that "has_topic" in some field of study.

For node sets "field_of_study" and "institution", a GNN on the full graph could produce meaningful hidden states for their few elements in the same way. However, in the sampled approach, it seems wasteful to do that from scratch for every subgraph. Instead, our model reads hidden states for them out of an embedding table. This way, the GNN can treat them as read-only nodes with outgoing edges only; the writing happens implicitly by gradient updates to their embeddings. (We choose to maintain a single embedding shared between the rounds of GNN updates.) – Notice how this modeling decision directly influences the sampling spec.

## Process Features

Usually in TensorFlow, the non-trainable transformations of the input features are split off into a `Dataset.map()` call while the main model consists of the trainable and accelerator-compatible parts. However, even this non-trainable part is put into a Keras model, which is a convenient way to track resources for export (such as lookup tables).

### Label Extraction

For node prediction, it is common to read out only one label per input graph, usually from the corresponding root node. In such cases, `tfgnn.keras.layers.ReadoutFirstNode` can be used instead of plain `Readout` to get it.
"""

def extract_labels(graphtensor: tfgnn.GraphTensor):
  labels = tfgnn.keras.layers.ReadoutFirstNode(
      node_set_name="paper",
      feature_name="labels")(graphtensor)
  graphtensor = graphtensor.remove_features(node_sets={"paper": ["labels"]})
  return graphtensor, labels

"""### Feature Preprocessing

Typically, feature preprocessing happens locally on nodes and edges. TF-GNN strives to reuse standard Keras implementations for this.  The `tfgnn.keras.layers.MapFeatures` layer lets you express feature transformations on the graph as a collection of feature transformations for the various graph pieces (node sets, edge sets, and context).
"""

# For nodes
def process_node_features(node_set: tfgnn.NodeSet, node_set_name: str):
  if node_set_name == "field_of_study":
    return {"hashed_id": tf.keras.layers.Hashing(50_000)(node_set["#id"])}
  if node_set_name == "institution":
    return {"hashed_id": tf.keras.layers.Hashing(6_500)(node_set["#id"])}
  if node_set_name == "paper":
    return {"feat": node_set["feat"]}
  if node_set_name == "author":
    return {"empty_state": tfgnn.keras.layers.MakeEmptyFeature()(node_set)}
  raise KeyError(f"Unexpected node_set_name='{node_set_name}'")

# For context and edges, in this example, we drop all features.
def drop_all_features(_, **unused_kwargs):
  return {}

# Combing feature mapping of context, edges and nodes
feature_mapping = tfgnn.keras.layers.MapFeatures(
    context_fn=drop_all_features,
    node_sets_fn=process_node_features,
    edge_sets_fn=drop_all_features)

# Combining the two feature pre-processing steps
feature_processors = [extract_labels, feature_mapping]

"""## Model Architecture

Typically, a model with a GNN architecture at its core consists of three parts:

1. The initialization of hidden states on nodes (and possibly also edges and/or the graph context) from their respective preprocessed features.
2. The core Graph Neural Network: several rounds of updating hidden states from neighboring items in the graph.
3. The readout of one or more hidden states into some prediction head, such as a linear classifier.

We are going to use one model for training, validation, and export for inference, so we need to build it from an input type spec with generic tensor shapes. (For TPUs, it's good enough to use it on a *dataset* with fixed-size elements.) Before defining the core Graph Neural Network, we show how to initialize the hidden states of all the necessary components (nodes, edges and context) given the pre-processed features.

## Initialization of Hidden States

The hidden states on nodes are created by mapping a dict of (preprocessed) features to fixed-size hidden states for nodes. Similarly to feature preprocessing, the `tfgnn.keras.layers.MapFeatures` layer lets you specify such a transformation as a callback function that transforms feature dicts, with GraphTensor mechanics taken off your shoulders.
"""

# Hyperparameters
paper_dim = 512

def set_initial_node_states(node_set: tfgnn.NodeSet, node_set_name: str):
  if node_set_name == "field_of_study":
    return tf.keras.layers.Embedding(50_000, 32)(node_set["hashed_id"])
  if node_set_name == "institution":
    return tf.keras.layers.Embedding(6_500, 16)(node_set["hashed_id"])
  if node_set_name == "paper":
    return tf.keras.layers.Dense(paper_dim)(node_set["feat"])
  if node_set_name == "author":
    return node_set["empty_state"]
  raise KeyError(f"Unexpected node_set_name='{node_set_name}'")

"""It is important to understand the distinction between feature pre-processing and hidden state intialization despite the fact that both of the steps are defined using `tfgnn.keras.layers.MapFeatures`. Feature pre-processing step is non-trainable and occurs asynchronous to the training loop. On the other hand, hidden state initialization is trainable and occurs on the corresponding accelerator.

## Core Graph Neural Networks Model

After the hidden states have been initialized, we pass the graph through the main Graph Neural Network model. The actual Graph Neural Network is a sequence of GraphUpdates. Each GraphUpdate inputs a GraphTensor and returns a GraphTensor with the same graph structure, but the hidden states of nodes have been updated using the information of the neighbor nodes. In our example, the input examples are sampled subgraphs with up to 4 hops, so we perform 4 rounds of graph updates which suffice to bring all information into the root node.  Here, we utilize the already-available [`VanillaMPNNGraphUpdate`](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/models/vanilla_mpnn/layers.py) to perform GraphUpdate. TF-GNN offers various modelling choices which are described in [tf-gnn-modeling-guide](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/gnn_modeling.md) and the [tf-gnn-models README](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/models/README.md).
"""

# Hyperparameters
l2_regularization = 6E-6
dropout_rate = 0.2
use_layer_normalization=True

def model_fn(graph_tensor_spec: tfgnn.GraphTensorSpec):
  graph = inputs = tf.keras.layers.Input(type_spec=graph_tensor_spec)
  graph = tfgnn.keras.layers.MapFeatures(
      node_sets_fn=set_initial_node_states)(graph)
  for _ in range(4):
    graph = vanilla_mpnn.VanillaMPNNGraphUpdate(
        units=128,
        message_dim=128,
        receiver_tag=tfgnn.SOURCE,
        l2_regularization=l2_regularization,
        dropout_rate=dropout_rate,
        use_layer_normalization=use_layer_normalization,
    )(graph)
  return tf.keras.Model(inputs, graph)

"""An important parameter to assign in the GraphUpdate function is the `receiver_tag`. To determine this tag, it is important to understand the difference between `tfgnn.SOURCE` and `tfgnn.TARGET`. *Source* indictates the node from where an edge originates while *Target* indicates the node to which an edge points to.

The graph sampler starts sampling from the root node (one can think of the root node as the main source of the subgraph) and stores edges in the direction of their discovery while sampling. Given this construct, the GNN needs to send information in the reverse direction towards the root. In other words, the information needs to be propagated towards the `SOURCE` of each edge, so that it can reach and update the hidden state of the root. Thus, we set the `receiver_tag` to be `tfgnn.SOURCE`. An interesting observation arising from the fact that `receiver_tag=tfgnn.SOURCE` is that since the node sets `"field_of_study"` and `"institution"` have no outgoing edge sets, the `VanillaMPNNGraphUpdate` does not change their hidden states: these remain the embedding tables from node state initialization.

## The Task

A Task collects the ancillary pieces for training a Keras model
with the graph learning objective. It also provides losses and metrics for that objective. Common implementations for classification and regression (by graph or root node) are provided in TF-GNN library.
"""

task = runner.RootNodeMulticlassClassification(
    node_set_name="paper",
    num_classes=349)

"""## The Trainer

A Trainer provides any training and validation loops. These may be uses of `tf.keras.Model.fit` or arbitrary custom training loops. The Trainer provides accesors to training properties (like its `tf.distribute.Strategy` and model_dir) and is expected to return a trained tf.keras.Model.

To keep this demo more interactive, we let Keras train and evaluate on fractions of a true epoch for a few minutes only. Of course the resulting accuracy will be poor. To fix that, feel free to edit the epoch_divisor according to your patience and ambition. ;-)
"""

# Hyperparameters
global_batch_size = 128
epochs = 5
initial_learning_rate = 0.001
epoch_divisor = 100  # To speed up the interactive demo_ds
steps_per_epoch = num_training_examples // global_batch_size // epoch_divisor
validation_steps = num_validation_examples // global_batch_size // epoch_divisor
learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, steps_per_epoch*epochs)
optimizer_fn = functools.partial(tf.keras.optimizers.Adam,
                                  learning_rate=learning_rate)

# Trainer
trainer = runner.KerasTrainer(
    strategy=strategy,
    model_dir="/tmp/gnn_model/",
    callbacks=None,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    restore_best_weights=False,
    checkpoint_every_n_steps="never",
    summarize_every_n_steps="never",
    backup_and_restore=False,
)

"""## Export options for inference

For inference, a SavedModel must be exported by the runner at the end of training. C++ inference environments like TF Serving do not support input of extension types like GraphTensor, so the `KerasModelExporter` exports the model with a SavedModel Signature that accepts a batch of serialized tf.Examples and preprocesses them like training did.

Note: After connecting this Colab to a TPU worker, explicit device placements are necessary to do the test on the colab host (which has the `/tmp/gnn_model` directory).
"""

save_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
model_exporter = runner.KerasModelExporter(options=save_options)

"""## Let the Runner do its magic!

Orchestration (a term for the composition, wiring and execution of the above abstractions) happens via a single run method with following signature shown below.

Training for 5 epochs of sampled subgraphs takes a few hours on a free colab with one GPU and should achieve an accuracy above 0.47. Lucky runs with many more epochs can reach 0.51. (Training with a Cloud TPU runtime is faster, but note that this set-up is not optimized specifically for TPUs.)
"""

runner.run(
    train_ds_provider=train_ds_provider,
    train_padding=train_padding,
    model_fn=model_fn,
    optimizer_fn=optimizer_fn,
    epochs=epochs,
    trainer=trainer,
    task=task,
    gtspec=example_input_graph_spec,
    global_batch_size=global_batch_size,
    model_exporters=[model_exporter],
    feature_processors=feature_processors,
    valid_ds_provider=valid_ds_provider, # <<< Remove if not training for real.
    valid_padding=valid_padding)

"""## Inference using Exported Model
At the end of training, a SavedModel is exported by the Runner for inference. For demonstration, let's call the exported model on the validation dataset from above, but without labels. We load it as a SavedModel, like TF Serving would. Analogous to the SaveOptions above, LoadOptions with a device placement are necessary when connecting this Colab to a TPU worker.





"""

# Inference on 10 examples
dataset = valid_ds_provider.get_dataset(tf.distribute.InputContext())
kwargs = {"examples": next(iter(dataset.batch(10)))}

load_options = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
saved_model = tf.saved_model.load(os.path.join(trainer.model_dir, "export"),
                                  options=load_options)
output = saved_model.signatures["serving_default"](**kwargs)

# Outputs are in the form of logits
logits = next(iter(output.values()))
probabilities = tf.math.softmax(logits).numpy()
classes = probabilities.argmax(axis=1)

# Print the predicted classes
for i, c in enumerate(classes):
  print(f"The predicted class for input {i} is {c:3} "
        f"with predicted probability {probabilities[i, c]:.4}")

"""## Next steps

This tutorial has shown how to solve a node classification problem in a large graph with TF-GNN using
  * the graph sampler tool to obtain manageable-sized inputs for each classification target,
  * the Runner for training GNNs with minimal coding.

The [Data Preparation and Sampling](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/data_prep.md) guide describes how you can create training data for other datasets.

The colab notebook [An in-depth look at TF-GNN](https://colab.research.google.com/github/tensorflow/gnn/blob/main/examples/notebooks/ogbn_mag_indepth.ipynb) solves OGBN-MAG again, but without the abstractions provided by the Runner and the ready-to-use VanillaMPNN model. Take a look if you like to know more, or want more control in designing GNNs for your own task.

For more complete documentation, please check out the [TF-GNN documentation](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/overview.md).
"""