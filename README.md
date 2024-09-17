# Evaluation Framework for Sepsis Treatment Strategies
Code release for Reproducible Evaluation Framework for Sepsis Treatment Strategies Learned via Offline Reinforcement Learning.

# Environment Setup
A Python environment may be constructed from `env.yml`:
```
conda env create -f env.yml
```

# MIMIC-III Dataset Setup
1. Request access, download, and install the [MIMIC-III dataset v1.4](https://physionet.org/content/mimiciii/1.4/). 
    * This should result in the installation of a MIMIC-III PostgreSQL database.
2. Either
    * [Disable PostgreSQL authentication](https://dba.stackexchange.com/questions/83164/postgresql-remove-password-requirement-for-user-postgres)
    * Reenable the password argument in `datasets/mimic_iii/preprocess_cohort.py` (lines 26 and 35).
3. Extract a patient cohort meeting the sepsis 3 criteria by running `datasets/mimic_iii/extract_cohort_from_postgres_db.py -u <your_postgres_user>`.
    * Replace <your_postgres_user> with the username you established when installing PostgreSQL.
    * This script should create several files in `datasets/mimic_iii/processed_files`.
4. Extract patient trajectories from the sepsis 3 cohort by running `datasets/mimic_iii/preprocess_cohort.py --out_dir datasets/mimic_iii/preprocessed_cohort --process_raw`.
    * This script will save two CSVs containing all patient trajectories to `datasets/mimic_iii/preprocessed_cohort`.
5. Create stratified splits by either
    * Running all cells in `notebooks/mimic-iii-splits.ipynb`.
        * Update your repository base path in cell 2 line 1 and cell 10 line 9.
        * Note: this will overwrite our splits.
    * Use our stratified splits
        * Skip this step (our splits are available to you in `datasets/mimic_iii_stratified_splits`.
6. Update dataset base paths in `utilities/mimic_iii_funcs.py` at lines
    * Line 14
    * Line 18
    * Line 39
    * Line 43

# Usage
### Continuous State Space Autoencoder
The code to train the state space sparse autoencoder may be found in `mimic_iii_train_autoencoder.py`.
To train the autoencoder:
1. Update the experiment base path (line 34).
2. Run `mimic_iii_train_autoencoder.py`.

### Behavior Policy
The code to learn the behavior policy via the KMeans SARSA method may be found in 'mimic_iii_learn_behavior_policy.py`. A behavior policy must be learned for each reward function you intend to evaluate. To learn behavior policies:
1. Update the experiment base path (line 33).
2. Update the reward function name (line 27).
    * See `mdp/reward_functions/factory.py` for a list of reward function options.
3. Run `mimic_iii_learn_behavior_policy.py`.

### Dueling DDQN
The code to train an Dueling DDQN agent (following [Raghu et al.'s approach](https://arxiv.org/abs/1705.08422)) may be found in `mimic_iii_d3qn.py`. To train an agent:
1. Update the experiment base path (line 70).
2. Update the MLflow state space run id (line 59).
    * This should be the run id assigned to the sparse autoencoder run you want to use.
    * The easiest way to identify the run id is through the MLflow UI (see https://mlflow.org/docs/latest/tracking.html).
    * For more information about MLflow see the [MLflow documentation](https://mlflow.org/docs/latest/introduction/index.html#).
3. Update the reward function name (line 61).
    * See `mdp/reward_functions/factory.py` for a list of reward function options.
4. Run `mimic_iii_d3qn.py`.

### Off-Policy Policy Evaluation (OPE)
The code to evaluate a sepsis treatment recommendation agent using OPE may be found in `mimic_iii_do_ope.py`. To evaluate an agent:
1. Update the experiment base path (line 90).
2. Update the MLflow state space run id (line 32).
    * See step 2 of Dueling DDQN for help identifying the run id.
4. Update the MLflow Dueling DDQN agent run id (line 35).
    * See step 2 of Dueling DDQN for help identifying the run id.
5. Update the behavior policy run id (line 33).
    * See step 2 of Dueling DDQN for help identifying the run id.
6. Update the reward function name (line 37).
7. [Optionally] Update the FQE run id to reuse FQE model(s) trained in a previous OPE evaluation run (line 49).
    * This will save time during evaluation if you're tuning MAGIC.
8. Run `mimic_iii_do_ope.py`.


# Acknowledgements
Our [Dueling DDQN](https://arxiv.org/abs/1705.08422) implementation was adapted from the following implementations:
- [Raghu et al.](https://github.com/aniruddhraghu/sepsisrl/blob/master/continuous/q_network.ipynb)
- [Phil Tabor](https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/DuelingDDQN/dueling_ddqn_agent.py)

Our [Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169) implementation was adapted from the following implementations:
- [Kostrikov et al.](https://github.com/ikostrikov/implicit_q_learning/blob/master/learner.py) (JAX)
- [Sebastian Dittert](https://github.com/BY571/Implicit-Q-Learning/blob/main/agent.py) (PyTorch)

Our [Model And Guided Importance sampling Combining (MAGIC)](https://arxiv.org/abs/1604.00923) implementation was adapted from [Voloshin et al.'s implementation](https://github.com/clvoloshin/COBS/blob/master/ope/algos/magic.py).

The script to extract a sepsis 3 patient cohort from the MIMIC-III dataset was originally written by [Komorowski et al.](https://github.com/matthieukomorowski/AI_Clinician) (MATLAB) and was adapted to Python by [Killian et al.](https://github.com/microsoft/mimic_sepsis/blob/main/preprocess.py)

The script to extract patient trajectories from a MIMIC-III sepsis 3 cohort was originally written by [Komorowski et al.](https://github.com/matthieukomorowski/AI_Clinician/blob/master/AIClinician_sepsis3_def_160219.m) and was adapted to Python by [Killian et al.](https://github.com/acneyouth1996/RL-for-sepsis-continuous/blob/yong_v0/scripts/sepsis_cohort.py)
