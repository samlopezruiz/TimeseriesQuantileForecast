# Multiobjective Framework for Quantile Forecasting in Financial Time Series using Transformers


Authors: Samuel López-Ruiz

### Abstract
> The uncertainty associated with the prediction is vital for decision making in financial time series forecasting. An attention-based deep learning model based on transformers is implemented to generate the high-performance quantile multi-horizon forecasting. The neural network model uses specialized components to analyze important variable in each forecasting task and visualize persistent temporal relationships. We use knowledge transfer to leverage the training process and only the weights from the deep learning model's last layer are optimized using a multi-objective evolutionary algorithm. The Pareto fronts obtained in this work, show that evolutionary algorithms can find wide range of solutions that allow the decision maker to fine tune the quantile forecasts. NSGA-II and NSGA-III are used as optimization algorithms and the example dataset consists of S&P 500 Futures.

## Model 
The forecasting model used in this work is presented
in the following paper: https://arxiv.org/pdf/1912.09363.pdf <br>
The model consists on a novel attention-based architecture which combines high-performance multi-horizon 
forecasting with interpretable insights into temporal dynamics. <br>
Its respective code can be found in: https://github.com/google-research/google-research/tree/master/tft
<br>
The quantile predictions obtained are plotted with the following nomenclature:
* _Target variable_:
exponential moving average (EMA) for the ES closing price in `red` color.
* _Mean quantile forecast_:
forecast for q=0.5 in `blue` color.

* _True target variable_:
ES closing price is the original target variable before the EMA smoothing and is shown in `magenta` color

* _Lower and upper quantiles_:
Quantile prediction interval is shown in `gray`.

* _Opacity_:
Predictions in the image are done with a 5 time step forecast horizon. The opacity in the prediction intervals (`gray`) and
mean prediction (`blue`) corresponds to the time step the prediction is made. Higher opacity means the prediction was
made fewer steps in the past.

<img src="https://github.com/samlopezruiz/TimeseriesQuantileForecast/blob/master/src/docs/TFTModel_ES_ema_r_q258_NSGA2_g100_p100_s0__tol5_all_pred_id31.png?raw=true" width="700" height="250"/>

## Code Organisation
This repository contains the source code for the Multi Objective Optimization of the quantiles forecasting for the 
Temporal Fusion Transformer model, along with the model code, training and evaluation routines.

The key configurations are defined as yaml files and are organised as:
* **download_datasets**: configuration to download relevant project datasets
* **volume_profile**: configuration to generate price volume profiles
* **hidden_markov_model**: configuration to generate the financial regime
* **split_dataset**: configuration to split the target dataset into train, test and valid subsets
* **additional_ds**: configuration to downsample additional datasets needed for the model training
* **preprocess**: configuration of the preprocessing for the data
* **model**: configuration of the model parameters (learning rate, batch size, etc)
* **vars_definition**: configuration variable definitions used in training.


The key project folders are organised as:

    .
    ├── src                     <- Source files
    │   ├── models              <- Implementation of models and algorithms
    │   └── timeseries          <- Model training and optimization
    │       ├── config          <- All configuration files needed
    │       ├── data_formatter  <- Gets dataset-specific column definitions
    │       ├── moo             <- Multi objective optimization scripts
    │       ├── plot            <- Plot functions
    │       ├── train_test      <- Model training and testing scripts
    │       ├── utils           <- Util functions
    │       └── volume          <- Volume profile visualizations
    ├── requirements.txt
    └── README.md


Additionally, inside the 'timeseries' folder, the following main scripts are listed according to 
their sequential execution and topic. 

* _Dataset definition_: 
  * (**download_ds.py**): downloads the datasets defined in the 'config' folder.
  * (**create_vol_profile.py**): creates the price volume profile indicator.
  * (**create_add_ds.py**): downsamples additional datasets needed for the training
  * (**create_regime.py**): detects the financial regime using a Hidden Markov Model (HMM).
* _Model training_: 
    * (**train_model.py**): trains TFT model
    * (**get_attention.py**): gets the attention variables from the trained model
    * (**plot_attention.py**): plots the attention obtained by the model
    * (**plot_forecasts.py**): plots the quantile forecasts
* _Model multi objective optimization_: 
    * (**moo_pareto_front.py**): gets the pareto front using the quantile coverage risk and quantile estimation risk as objectives
    * (**moo_test_model.py**): uses a solution from the pareto front to generate the predictions


The data, trained models, results, images and forecasts are saved in the 'output' folder. 

## Running experiement
The running project consists of S&P futures index dataset complimented with Nasdaq and other
financial timeseries. <br>
To implement another project, change the line``project='snp'`` in the main scripts and 
replicate the configuration yaml files.

### Step 0: Clone repo and install requirements
   ```sh
   git clone https://github.com/samlopezruiz/TimeseriesQuantileForecast
   cd TimeseriesQuantileForecast
   pip install requirements.txt
   ```
   
### Step 1: Download and prepare data
To download the experiment data, run the following script:
```bash
python3 src/timeseries/download_ds.py
```
Execute the following scripts to prepare all datasets needed
```bash
python3 src/timeseries/create_vol_profile.py
python3 src/timeseries/create_add_ds.py
python3 src/timeseries/create_regime.py
```
Finally, to split the target dataset, run:
```bash
python3 src/timeseries/split_ds.py
```

### Step 2: Train and evaluate network
To train the network with the default parameters, run:
```bash
python3 src/timeseries/train_model.py
```
To plot the forecasts, run:
```bash
python3 src/timeseries/plot_forecasts.py
```
Finally, to get and plot the attention obtained, run:
```bash
python3 src/timeseries/get_attention.py
python3 src/timeseries/plot_attention.py
```

### Step 3: Multi objective optimization
To get the pareto front, run:
```bash
python3 src/timeseries/moo_pareto_front.py
```
To test a solution from the pareto front, run:
```bash
python3 src/timeseries/moo_test_model.py
```



## Customising scripts for new datasets
### Step 1: Datasets
Configure `download_datsets.yaml` to include the dataset download url.

```yaml
# dataset subfolder
day:

    # dataset file
    - description: S&P dataset with 1 day frequency. ene 2012 - june 2021
      file_name: ES_day_2021-2021_6.csv
      url: https://mega.nz/file/PYtEQSKJ#iCAd42fywRakQeTzVx6LqLzFbn3q8nndX4wul_eqzUc
```
Configure `split_dataset.yaml` to select the respective target data and the split configuration.
The dataset is divided in time subsets and then splitted into train, test and valid subsets. The dataset can 
also be downsampled and trimmed if needed. The following image shows an example of the subsets shown with different colors:

<img src="https://github.com/samlopezruiz/TimeseriesQuantileForecast/blob/master/src/docs/split_ES_minute_5T_dwn_smpl_2015-01_to_2021-06_g12week_r15.png?raw=true" width="400" height="200" />

```yaml
data_cfg:
    inst: ES
    subfolder: minute
    filename: ES_min_2021-2021_6.csv
    trim_data_from: 2015-01
    trim_data_to: 2021-06
    # specify additional datasets (if any)
    append_datasets:
        - filename: Vol_5levels_ESc_ES_vol_2021-2021_6.z
          path: ['vol_profile'] # specify path as list
    downsample: True
    downsample_p: 60T

split_cfg:
    # group by year, week, day, or hour
    group: week
    groups_of: 8
    test_ratio: 0.15
    valid_ratio: 0.15
    random: True
    time_thold:
        days:
        hours: 3
        minutes:
        seconds:
    test_time_start: (8, 30)
    test_time_end: (15, 0)
    time_delta_split: True
```

### Step 2: Model params
Modify `model/q###.yaml` to specify the model parameters.

```yaml
fixed_params:
    quantiles: [0.3, 0.5, 0.7]
    num_epochs: 100
    early_stopping_patience: 5
    multiprocessing_workers: 12

model_params:
    total_time_steps: 53
    num_encoder_steps: 48
    dropout_rate: 0.3
    hidden_layer_size: 16
    learning_rate: 0.01
    minibatch_size: 64
    max_gradient_norm: 0.01
    num_heads: 4
    stack_size: 1
```

### Step 3: Dataset definition and preprocessing

Modify `preprocess/config_file.yaml` to specify the preprocessing of the dataset. The 
configuration file allows the preprocessing of the `target_dataset` (where the target variable is located)
and additional datasets that might be useful for the forecasting. 

Tthe following parameters are available for preprocessing:
* _macd_vars_: list of variable to calculate the MACD indicator
* _rsi_vars_: list of variable to calculate the RSI indicator
* _macd_periods_: list of the fast periods of the MACD
* _returns_vars_: list of variables for which the returns will be calculated
* _returns_from_ema_: list of variables for which the returns will be calculated
* _returns_vars_: calculates the returns from the EMA(price, period)
* _true_target_: specify when the true target is different from the target used for forecasting,  e.g. the price might be the true target, but the returns are used as target variable in forecasting

```yaml
# main dataset
target_dataset:
  filename: split_ES_minute_60T_dwn_smpl_2015_1_to_2021_6_g8week_r15
  subfolder: ['split']
  macd_vars: ['ESc']
  rsi_vars: ['ESc']
  macd_periods: [24, 12, 6]
  returns_vars: ['ESc', 'ESh', 'ESo', 'ESl']
  returns_from_ema: True,
  ema_period: 3
  true_target: ESc_e3

# append additional datasets features
additional_datasets:
  - filename: regime_ESc_r_T10Y2Y_VIX_2021_6_to_2021_6
    subfolder: ['regime']
    use_only_vars: ['state']
```

Finally, specify a``vars_definition/definition.yaml`` that defines how the variables are used within 
the TFT model. The `columnDefinition` specifies the list of variables with their respective dataType, and inputType.
The `additionalDefinitions` are later appended to the `columnDefinitions` and usually are variables reused 
in several variable configurations. 

```yaml
columnDefinition:

    - varName: ESc_e3_r
      dataType: REAL_VALUED
      inputType: TARGET
      
    - varName: datetime
      dataType: DATE
      inputType: TIME

additionalDefinitions:
  - filename: append_known_date
  - filename: append_ES_vars
```

## Results
The following two images show the variable selection for the running example (left) and the attention according the
position index for the first transformer head (right).

<p float="left">
<img src="https://github.com/samlopezruiz/TimeseriesQuantileForecast/blob/master/src/docs/q159_hist_attn.png?raw=true" width="300" height="200" />
<img  src="https://github.com/samlopezruiz/TimeseriesQuantileForecast/blob/master/src/docs/hist_attn_position_head.png?raw=true" width="400" height="200" />
</p>


The result from the multi objective optimization consists of the Pareto front with the _quantile coverage risk_ and _quantile estimation risk_ as
the objectives. The left images shows the pareto front found for three configurations of quantiles and the
right image shows the pareto front and a selected solution (in red) inside a tolerance window defined by a
threshold increment in the total error and which can later be used to generate forecasts. 

<p float="left">
<img src="https://github.com/samlopezruiz/TimeseriesQuantileForecast/blob/master/src/docs/vary_quantiles_ES_ema_r_moo_results.png?raw=true" width="350" height="350" />
<img src="https://github.com/samlopezruiz/TimeseriesQuantileForecast/blob/master/src/docs/TFTModel_ES_ema_r_q258_NSGA2_g100_p100_s0_lix33_uix31_tol5_pf.png?raw=true" width="350" height="350" />
</p>


The forecast of the solution marked in red is shown in the following image. It can be observed that the larger prediction
intervals have increased with respect to the forecast shown previously and therefore the _quantile coverage risk_ is reduced.

<img src="https://github.com/samlopezruiz/TimeseriesQuantileForecast/blob/master/src/docs/TFTModel_ES_ema_r_q258_NSGA2_g100_p100_s0_lix33_uix31_tol5_all_pred_id31.png?raw=true" width="700" height="250"/>

