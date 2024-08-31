# Decoding Finger Velocity from Cortical Spike Trains with Recurrent Spiking Neural Networks

This repository contains the code associated with the publication:

    Liu, T.*, Gygax, J.*, Rossbroich, J.*, Chua, Y., Zhang, S. and Zenke, F. "Decoding Finger Velocity from Cortical Spike Trains with Recurrent Spiking Neural Networks". arXiv preprint

Click [here](PLACEHOLDER) to access the preprint.

This data was also submitted to the [IEEE BioCAS 2024 Grand Challenge on Neural Decoding for Motor Control of non-Human Primates](https://ieee-dataport.org/competitions/ieee-biocas-2024-grand-challenge-neural-decoding-motor-control-non-human-primates) under the team name **ZenkeLab-QAAS**, where it achieved the second-highest overall decoding performance and was determined the winner of the **Best Trade-off between R2 score and Solution Complexity** track.

## About this submission

We have constructed two spiking neural network (SNN) models for the decoding of hand kinematics from neural activity data. The two models serve different purposes:

- `bigRSNN`: a large, high-performance recurrent SNN model designed to probe the upper limits of decoding performance possible with recurrent SNNs.
- `tinyRSNN`: a small, resource-efficient recurrent SNN model designed to meet the strict energy and computational constraints of real-time applications in brain-machine interfaces. 

## Model Descriptions

All models were implemented using the [`stork`](https://github.com/fmi-basel/stork) library for training spiking neural networks and trained using surrogate gradients. Hyperparameters were optimized based on the average validation set performance across all sessions.

### bigRSNN 

The `bigRSNN` model was designed to maximize the R2 score on the validation set regardless of the computational resources required.
The model consists of a single recurrent spiking neural network (SNN) layer with 1024 LIF neurons. The input size corresponds to the number of electrode channels for each monkey. The readout layer consists of five independent readout heads with 2 leaky integrator readout units each. 
The final output for X and Y coordinates is obtained by averaging the predictions of the five readout heads. 
Synaptic and membrane time constants are heteregeneous for each hidden and readout unit and were optimized during training.

### tinyRSNN 

The `tinyRSNN` model was designed to achieve a good trade-off between R2 score and computational complexity. 
It consists of a single recurrent spiking neural network (SNN) layer with 64 LIF neurons. 
The input layer size matches the number of electrode channels for each monkey.
The readout layer consists of 2 leaky integrator units, one each for the X and Y coordinates. 
As in the `bigRSNN` model, synaptic and membrane time constants are heteregeneous for each hidden and readout unit and were optimized during training. 

To further reduce the computational complexity of the `tinyRSNN` model, we applied an additional activity regularization loss acting on hidden layer spike trains during training, which penalizes firing rates above 10 Hz. 
To enforce connection sparsity, we implemented an iterative pruning strategy of synaptic weights during training. 
At each iteration of the pruning procedure, the $N$ smallest synaptic weights in each weight matrix were set to zero and the network was re-trained.
Finally, the model is set to half-precision floating point format after training to reduce the memory footprint and speed up inference.

## Organization
The code is organized as follows:

- `/challenge`: contains source code for data loaders, models, training and evaluation
- `/conf`: contains configuration files for training and evaluation scripts (uses the [`hydra`](https://github.com/facebookresearch/hydra) framework)
- `/models`: contains model state dictionaries for the best models obtained during training, with the format: `/models/session_name/model_name-rand_seed.pth`
- `/results`: contains evaluation results for each model & session independently. Each `.json` file summarizes model performance across five random seeds.

#### Training and evaluation scripts

The scripts used for training the submitted models are `train-bigRSNN.py` and `train-tinyRSNN.py`. The evaluation script used to run [NeuroBench](https://github.com/NeuroBench/neurobench) benchmarks is `evaluate.py`. Configuration files for these scripts are located in the `/conf` directory.

#### Results
The files `results_summary_bigRSNN.json` and `results_summary_tinyRSNN.json` hold a summary of the results as submitted to the [IEEE BioCAS 2024 Grand Challenge on Neural Decoding for Motor Control of non-Human Primates](https://ieee-dataport.org/competitions/ieee-biocas-2024-grand-challenge-neural-decoding-motor-control-non-human-primates) (averaged across five random seeds). For results corresponding to individual seeds, please refer to the `/results` folder.


## Installation

We used Python 3.10.12 for the development of this code. To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Because `stork` is under continuous development, we recommend installing the exact commit used for this submission, as indicated in the `requirements.txt` file.

## Reproducing the results

To reproduce the results, first go to the `/conf/data/data-default.yaml` file and set the `data_dir` parameter to the path of the data directory containing the challenge data. Because models are pre-trained on all publicly available sessions where the same number of electrodes were used for each monkey, the data directory should contain all files from the ["Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology" dataset](https://zenodo.org/records/583331). For only pretraining on the sessions used in the [IEEE BioCAS 2024 Grand Challenge on Neural Decoding for Motor Control of non-Human Primates](https://ieee-dataport.org/competitions/ieee-biocas-2024-grand-challenge-neural-decoding-motor-control-non-human-primates), set **data.pretrain_filenames=challenge-data**.

Second, set the output directory for `hydra` in the `/conf/config.yaml` file to the desired output directory. Each run of the training script will create a new subdirectory in the output directory to store configuration files, logs, results, plots, and a copy of model state dictionaries. This defaults to `./outputs`.

### Training

To train the models from scratch, run the following commands:

```bash
python train-bigRSNN.py --multirun seed=1,2,3,4,5
```

and

```bash
python train-tinyRSNN.py --multirun seed=1,2,3,4,5
```

This will train the models on the training set with the specified random seeds and overwrite the model state dicts in the `/models` directory. 

To train only one model with a specific seed, run the following commands:

```bash
python train-bigRSNN.py seed=1
```

and

```bash
python train-tinyRSNN.py seed=1
```

By default, benchmarking is run after training and results are recorded in the `hydra` generated output log file. 
To obtain a summary of the results from the log file, without re-running evaluation (see below), please refer to the `results_extract_from_logs.ipynb` notebook.

**Note**: Training the models from scratch requires a GPU and significant computational resources. Training the `bigRSNN` model with one initial random seed (pre-training & fine-tuning on each session) takes approximately 12 hours on an NVIDIA RTX A4000 GPU. Training the `tinyRSNN` model with one initial random seed (pre-training & fine-tuning on each session) takes approximately 6 hours on an NVIDIA RTX A4000 GPU.

### Evaluation and benchmarking

We supplied the model state dictionaries for the best models obtained during training in the `/models` directory. Models are sorted into subdirectories by session and monkey:

- `/loco01`: models trained on session `loco_20170210_03`
- `/loco02`: models trained on session `loco_20170215_02`
- `/loco03`: models trained on session `loco_20170301_05`
- `/indy01`: models trained on session `indy_20160622_01`
- `/indy02`: models trained on session `indy_20160630_01`
- `/indy03`: models trained on session `indy_20170131_02`

There are five `tinyRSNN` models and five `bigRSNN` models for each session, corresponding to five different initializations. To evaluate these models on the test set, run the following commands:

```bash
python evaluate.py modelname=bigRSNN
```

and

```bash
python evaluate.py modelname=tinyRSNN
```

By default, the evaluation script uses a custom wrapper for stork models to be compatible with the [NeuroBench](https://github.com/NeuroBench/neurobench) benchmarking suite (see code in the `/challenge/neurobench` directory). Alternatively, the user can set the `use_snnTorch_model` flag to `True`, to convert the original `stork` model to an equivalent model using the [snnTorch](https://github.com/jeshraghian/snntorch) library and run evaluation using the unmodified Neurobench code, which leads equivalent results.

The evaluation scripts saves results in the `json` format. Results for individual models and sessions are saved in the `/results` folder. Additionally, summaries for each model are saved in the root directory.

## Results

For convenience, the following tables display the results obtained from the `tinyRSNN` and `bigRSNN` networks, as well as the two baseline spiking networks trained on the same datasets provided in the [NeuroBench](https://github.com/NeuroBench/neurobench) codebase.

### Table 1: Model performance across sessions (R2 values)

| Session | [Baseline SNN2](https://github.com/NeuroBench/neurobench/blob/main/neurobench/examples/primate_reaching/benchmark_SNN2.py)  | [Baseline SNN3](https://github.com/NeuroBench/neurobench/blob/main/neurobench/examples/primate_reaching/benchmark_snn_3.py)  | tinyRSNN               | bigRSNN               |
|---------|-------|-------|------------------------|-----------------------|
| I1      | 0.677 | 0.697 | 0.752 ± 0.003          | 0.770 ± 0.003         |
| I2      | 0.501 | 0.577 | 0.545 ± 0.004          | 0.585 ± 0.012         |
| I3      | 0.599 | 0.652 | 0.746 ± 0.007          | 0.772 ± 0.006         |
| L1      | 0.571 | 0.623 | 0.622 ± 0.003          | 0.698 ± 0.006         |
| L2      | 0.515 | 0.568 | 0.608 ± 0.006          | 0.629 ± 0.008         |
| L3      | 0.620 | 0.681 | 0.690 ± 0.006          | 0.734 ± 0.005         |
| **Mean**| **0.581** | **0.633** | **0.660 ± 0.076** | **0.698 ± 0.070** |


### Table 2: Computational demand across sessions

| Model      | Session | Footprint  | Connection Sparsity    | Activation Sparsity    | Dense    | Eff. MACs | Eff. ACs                   |
|------------|---------|------------|------------------------|------------------------|----------|-----------|----------------------------|
| **tinyRSNN** | I1      | 21000      | 0.47 ± 0.02         | 0.9838 ± 0.0004         | 10368    | 0         | 299 ± 14          |
|            | I2      | 21000      | 0.45 ± 0.03         | 0.9853 ± 0.0003         | 10368    | 0         | 197 ± 12          |
|            | I3      | 21000      | 0.50 ± 0.00         | 0.9842 ± 0.0002         | 10368    | 0         | 143 ± 2           |
|            | L1      | 33288      | 0.44 ± 0.04         | 0.9831 ± 0.0001         | 16512    | 0         | 354 ± 28          |
|            | L2      | 33288      | 0.42 ± 0.02         | 0.9832 ± 0.0002         | 16512    | 0         | 405 ± 20          |
|            | L3      | 33288      | 0.45 ± 0.03         | 0.9820 ± 0.0003         | 16512    | 0         | 427 ± 28          |
|            | **mean** | **27144**  | **0.45 ± 0.04**     | **0.9836 ± 0.0011**     | **13440**| **0**  | **213 ± 66**      |
| **bigRSNN** | I1      | 4636752    | 0.0                 | 0.9622 ± 0.0002         | 1157120  | 0         | 48097 ± 254        |
|            | I2      | 4636752    | 0.0                 | 0.9718 ± 0.0006         | 1157120  | 0         | 34837 ± 659        |
|            | I3      | 4636752    | 0.0               | 0.9721 ± 0.0002         | 1157120  | 0         | 33289 ± 181        |
|            | L1      | 5029968    | 0.0                 | 0.9677 ± 0.0012         | 1255424  | 0         | 44664 ± 1258       |
|            | L2      | 5029968    | 0.0                 | 0.9674 ± 0.0010         | 1255424  | 0         | 45945 ± 1097       |
|            | L3      | 5029968    | 0.0                 | 0.9686 ± 0.0014         | 1255424  | 0         | 45189 ± 1437      |
|            | **mean** | **4833360**| **0.0**     | **0.9683 ± 0.0034**     | **1206272**| **0** | **45266 ± 1376**    |
| **[Baseline SNN2](https://github.com/NeuroBench/neurobench/blob/main/neurobench/examples/primate_reaching/benchmark_SNN2.py)** | **mean**       | **29248**  | **0.0**             | **0.9967**              | **7300** | **0**  | **414**               |
| **[Baseline SNN3](https://github.com/NeuroBench/neurobench/blob/main/neurobench/examples/primate_reaching/benchmark_snn_3.py)** | **mean**       | **33996**  | **0.0**             | **0.7880**              | **43680**| **3226** | **5831**          |
