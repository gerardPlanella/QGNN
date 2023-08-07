# ToGePi: Topology and Geometry informed Positional Information

## Project Description

Graph neural networks (GNNs) have emerged as the dominant learning architectures for graph data. Among them, Equivariant Graph Neural Networks (EGNNs) introduced a novel approach to incorporate geometric information, ensuring equivariance throughout the system. However, the EGNN architecture has two main limitations. Firstly, it underutilizes the topological information inherent in the graph structure, and secondly, achieving SOTA performance necessitates a fully connected graph, which may not always be feasible in certain applications. In addition, the Learnable structural and Positional Encodings (LSPE) framework proposes to decouple structural and positional representations to learn better these two essential properties by using implicit topological information. In this work, we investigate the extent to which structural encodings in geometric methods contribute in capturing topological information. Furthermore, inspired by Equivariant Message Passing Simplicial Network (EMPSN) architecture, which integrates geometric and topological information on simplicial complexes, we introduce an approach that leverages geometry to enhance positional encodings within the LSPE framework. We empirically show through our proposed method that conditioning the learnable PEs with the absolute distance between particles (for the QM9 dataset) can be beneficial to learn better representations, given that the model has sufficient complexity. Our method exhibits promising potential for graph datasets with limited connectivity, offering opportunities for advantageous outcomes by effectively handling situations where achieving a fully connected graph is not feasible.


---

## Setting up the Environment
In order to set up the environment for reproducing our experiments, 
install the appropriate conda environment that suits your hardware specifications. 
We put forward two YAML environment files: `environment_gpu.yml` CUDA support and `environment.yml` for CPU (and MPS) support.

```commandline
$ conda env create -f <environment_filename>
```

## Downloading the Data
In all of our experiments, we use the QM9 dataset, first introduced by [Ramakrishnan et al., 2014](https://www.nature.com/articles/sdata201422), comprises approximately 130,000 graphs, each
consisting of around 18 nodes. The objective of analyzing this dataset is to predict 13 quantum chemical properties.
Nevertheless, this study only focuses on inferring the Isotropic Polarizability $\alpha$.

The datasets are automatically downloaded once an experiment 
is run with a specific argument configuration for arguments `--dataset` and `--pe` and `--pe_dim`. 
The arguments can take on the following values:

| Dataset | Explanation                                     |
|---------|-------------------------------------------------|
| QM9     | The original QM9 dataset. **Default**           |  
| QM9_fc  | The fully-connected variant of the QM9 dataset. |

| Positional Encoding | Explanation                                                                                             |
|---------------------|---------------------------------------------------------------------------------------------------------|
| nope                | The dataset is initialized with no PE concatenated to the hidden node state. **Default**                |  
| rw                  | The dataset is initialized with Random-Walk PE concatenated to the hidden node state.                   |
| lap                 | The dataset is initialized with a Laplacian Eigenvector-based PE concatenated to the hidden node state. |

| PE Dimension | Explanation                                              |
|--------------|----------------------------------------------------------|
| [1-28]       | The dimension of the PE vectors per node. **Default 24** |

## Reproducibility of Experiments
We use [WandB](https://wandb.ai) as our central dashboard to keep track of your hyperparameters, system metrics, and predictions and results.
Before running the experiments, login to your wandb account by entering the following command:
```commandline
$ wandb login 
```

For reproducing the experiments, run the following commands in the terminal after activating your environment.

```commandline
$ python main.py --config mpnn_1.json
```

The training and network parameters for each experiment is stored in a json file in the config/ directory. 
The full path of the config file is not necessary.

Alternatively, instead of the config argument, one can start runs by specifying each individual run argument. For example:

```commandline
python main.py --model mpnn --pe rw --pe_dim 24 --include_dist --lspe
```

One can additionally pass another argument `--write_config_to <new_config_filename>` to write the argument configuration to a file for later convenience when running multiple experiments.
All the running arguments alongside their explanation can be found under `main.py`.

### Output, checkpoints and visualizations

Output results and visualisations are processed directly to WandB, and are accessible [here](https://api.wandb.ai/links/dl2-gnn-lspe/krcsymc6).  
The saved model weights are stored under `saved_models`. We acknowledge that not anybody might have access to the required computational 
resources to train each of the models we tested, and thus we provide the saved model weights in the HuggingFace repository [here](https://huggingface.co/datasets/lucapantea/egnn-lspe).
See [demos/main.ipynb](demos/main.ipynb) for an overview of how to load the model weights and evaluate a given odel configuration.   
