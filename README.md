# Towards Quantum Graph Neural Networks

## Master Thesis by Gerard Planella Fontanillas

This repository contains the code and data associated with the master thesis "Towards Quantum Graph Neural Networks," which explores the potential of integrating quantum physics knowledge into Graph Neural Networks (GNNs) to address the computational challenges posed by quantum many-body systems.

## Abstract

Quantum many-body systems pose significant computational challenges due to the exponential growth of their state spaces with system size. Traditional methods, such as Tensor Network (TNs) algorithms like Density Matrix Renormalisation Group (DMRG) and Simple Update (SU), have proven to be very effective in modelling these highly complex systems. However,  they have been found to suffer scalability and complexity issues. This thesis explores the potential of imbuing  Graph Neural Networks (GNNs) with quantum physics knowledge to address these challenges. 

We develop and evaluate several supervised learning models, including BlochGNN, EBlochGNN, RDMNet, and Neural Enhanced Quantum Belief Propagation (NEQBP), as well as an unsupervised Quantum Belief Propagation (QBP) method and a semi-supervised QBP-EBlochGNN model. These models are tested on diverse datasets, including Matrix Product States (MPS) and Projected Entangled Pair States (PEPS) of up to 20 qubits. Our research findings indicate that among the supervised learning methods, EBlochGNN performs the best but still does not surpass DMRG and only outperforms SU in one dataset. On the other hand, the unsupervised QBP method outperforms the baselines in all datasets. However, it is important to note that further improvements are needed in the implementation of QBP, as it currently faces issues related to efficiency. This work highlights the importance of integrating physics knowledge into AI models, setting the stage for future research to improve the accuracy and scalability of these approaches for quantum many-body systems.

## Repository Structure

```
├── .gitignore                     # Specifies files and directories to be ignored by git
├── .vscode/launch.json            # Configuration for Visual Studio Code debugger
├── LICENSE                        # License information
├── README.md                      # This readme file
├── baselines/                     # Baseline models and scripts
│   ├── __init__.py
│   ├── beliefpropagation/         # Belief propagation algorithms and notebooks
│   │   ├── belief_prop.ipynb
│   │   ├── factor.py
│   │   ├── simple_update.py
│   │   ├── tn_simple_update.ipynb
│   │   ├── variable_elimination.ipynb
│   ├── gnn.py                     # GNN baseline implementation
│   ├── tensorNetworks.py          # Tensor network utilities
├── config/                        # Configuration files for various models and experiments
│   ├── dmrg.json
│   ├── fu.json
│   ├── gnn.json
│   ├── qbp.json
│   ├── qbp_by_parts.json
│   ├── qgnn_1.json
│   ├── qgnn_2.json
│   ├── qgnn_2_rdm.json
│   ├── qgnn_3.json
│   ├── qgnn_3_rdm.json
│   ├── qgnn_4.json
│   ├── qgnn_4_aux.json
│   ├── qgnn_4_it.json
│   ├── su.json
│   ├── su_gen.json
├── dataset/                       # Dataset generation and loading scripts
│   ├── __init__.py
│   ├── ising/                     # Ising model datasets
│   │   ├── dataset_generation.ipynb
│   │   ├── generate_dataset.py
│   │   ├── isingModel.py
│   ├── qm9/                       # QM9 dataset utilities
│   │   ├── qm9.py
├── environment_qgnn_gpu.yml       # Conda environment configuration file for GPU
├── eval_model_output.py           # Script to evaluate model outputs
├── jobs/                          # Job scripts for various training and dataset creation tasks
│   ├── qbp_by_parts/              # Job scripts for QBP by parts experiments
│   │   ├── run_train_qbp1.job
│   │   ├── run_train_qbp2.job
│   │   ├── run_train_qbp3.job
│   │   ├── run_train_qbp4.job
│   │   ├── run_train_qbp5.job
│   │   ├── run_train_qbp6.job
│   ├── qgnn4/                     # Job scripts for QGNN4 experiments
│   │   ├── run_train_qgnn_4_10x1.job
│   │   ├── run_train_qgnn_4_20x1.job
│   │   ├── run_train_qgnn_4_3x3.job
│   │   ├── run_train_qgnn_4_5x4.job
│   ├── qgnn4_it/                  # Job scripts for QGNN4 iterative experiments
│   │   ├── run_train_qgnn_4_1.job
│   │   ├── run_train_qgnn_4_2.job
│   │   ├── run_train_qgnn_4_3.job
│   │   ├── run_train_qgnn_4_4.job
│   ├── rdms/                      # Job scripts for RDM-based QGNN experiments
│   │   ├── run_train_qgnn_2_10x1.job
│   │   ├── run_train_qgnn_2_20x1.job
│   │   ├── run_train_qgnn_2_3x3.job
│   │   ├── run_train_qgnn_2_5x4.job
│   │   ├── run_train_qgnn_3_10x1.job
│   │   ├── run_train_qgnn_3_20x1.job
│   │   ├── run_train_qgnn_3_3x3.job
│   │   ├── run_train_qgnn_3_5x4.job
│   ├── run_create_dataset.job     # Job script for creating datasets
│   ├── run_create_dataset_55.job
│   ├── run_install_env.job        # Job script for installing environment
│   ├── run_train_dmrg.job         # Job script for training DMRG
│   ├── run_train_fu.job           # Job script for training FU
│   ├── run_train_qbp.job          # Job script for training QBP
│   ├── run_train_qgnn_1.job       # Job script for training QGNN1
│   ├── run_train_qgnn_2.job       # Job script for training QGNN2
│   ├── run_train_qgnn_3.job       # Job script for training QGNN3
│   ├── run_train_qgnn_4.job       # Job script for training QGNN4
│   ├── run_train_su.job           # Job script for training SU
│   ├── run_train_su_gen.job       # Job script for training SU general
├── lib/                           # Utility functions and classes
│   ├── __init__.py
│   ├── agg.py                     # Aggregation utilities
│   ├── pe.py                      # Positional encoding utilities
│   ├── rdm.py                     # RDM utilities
│   ├── utils.py                   # General utilities
├── main.py                        # Main script for running experiments
├── qbp_gnn/                       # QBP-GNN models and training scripts
│   ├── __init__.py
│   ├── qbp copy.py
│   ├── qbp.py
│   ├── qbp_train.py
│   ├── qgnn_bp.py
│   ├── qgnn_bp_with_z.py
│   ├── qgnn_train.py
│   ├── tensor_network.py
├── qgnn/                          # QGNN models and scripts
│   ├── Readme.md
│   ├── __init__.py
│   ├── qgnn.py
│   ├── qgnn_2.py
│   ├── qgnn_bidir.py
│   ├── qgnn_em.py
│   ├── qgnn_em_R.py
├── split_dataset.py               # Script to split datasets
├── train.py                       # Script to train models
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Conda
- CUDA-compatible GPU (optional but recommended)

### Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/username/repo.git
    cd repo
    ```

2. Create and activate a Conda environment:
    ```sh
    conda env create -f environment_qgnn_gpu.yml
    conda activate qgnn_gpu
    ```

### Usage

1. **Data Preparation**:
    - Ensure that the datasets are placed in the `dataset/` directory.

2. **Training and Evaluating Models**:
    - Execute the main script to run experiments and evaluations. Use the `-h` parameter to display all available options and configurations:
        ```sh
        python main.py -h
        ```

## Results

The results of our experiments are logged to WandB if the `--no-wandb` flag is not set. Otherwise, the results are shown through matplotlib.

### Setting up WandB

1. Sign up for an account on [Weights & Biases](https://wandb.ai/).
2. Find your API key in your [WandB account settings](https://wandb.ai/settings).
3. Set the API key in your environment:
    ```sh
    export WANDB_API_KEY=your_api_key_here
    ```

## Contributing

We welcome contributions to improve this project. If you have any suggestions or find any issues, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We thank our supervisors, Dr. Max Welling and Dr. Vikas Garg, for their guidance and support. Moreover, we also thank Adrian Muller and Floor Eijkelboom for their help during the project. This work was conducted as part of the MSc in Artificial Intelligence program at the University of Amsterdam.

---

For more detailed information, please refer to the [full thesis document](AI_MSc_Thesis_Gerard_Planella.pdf).
