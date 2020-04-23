# Standard library imports
from typing import Optional, List, Dict, Tuple
from glob import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Third-party imports
import mxnet as mx
from mxnet.gluon import HybridBlock
from gluonts.trainer import Trainer
from gluonts.distribution import NegativeBinomialOutput
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import GluonPredictor
import pmdarima as pm

# First Party imports
from .rec_penalty import DeepARRecPenaltyEstimator

def fit_deepar(
    training_data: ListDataset,
    validation_data: ListDataset = None,
    freq: str = 'M',
    pred_length: int = 12,
    num_layers: int = 2,
    hidden_dim: int = 40,
    use_cat_var: bool = False,
    cardinality: Optional[List[int]] = None,
    epochs: int = 25,
    batch_size: int = 32,
    self_supervised_penalty: float = 0.0,
    embedding_agg_penalty: float = 0,
    embedding_dist_metric: str = 'cosine',
    embedding_dim_ratio: float = 1,
    hierarchy_agg_dict: Optional[Dict[int, List[int]]] = None,
    ignore_future_targets: bool = False,
    print_rec_penalty: bool = True,
    tb_log_dir: str = None,
) -> Tuple[GluonPredictor, HybridBlock]:
    """ fits DeepAREstimator with optional reconciliation penalties to training dataset

    Arguments:
        training_data {ListDataset} -- training data

    Keyword Arguments:
        validation_data {ListDataset} -- optional validation data. If set, the model checkpoint
            from the best epoch with respect to this dataset will be returned
        freq {str} -- frequency (default: {'M'})
        pred_length {int} -- prediction length (default: {12})
        num_layers {int} -- number of RNN layers to include in estimator (default: {2})
        hidden_dim {int} -- dimension of hidden state of each RNN (default: {40})
        use_cat_var {bool} -- whether to include the whether to include a categorical variable for
            series in this model (default: {False})
        cardinality {Optional[List[int]]} -- cardinality of each categorical variable if including 
            in model (default: {None})
        epochs {int} -- number of training epochs (default: {25})
        batch_size {int} -- if self-supervised reconciliation penalty > 0, will be set to training set size
            (default: {32})
        self_supervised_penalty {float} -- lambda for self-supervised reconciliation penalty 
            (default: {0.0})
        embedding_agg_penalty {float} -- lambda for embedding rec. penalty 
            (default: {0.0})
        embedding_dist_metric {str} -- distance metric for embedding rec. penalty
            (default: {'cosine'})
        embedding_dim_ratio {float} -- ratio between embedding dim and RNN hidden state dim
            (default: {1.0})
        hierarchy_agg_dict {Optional[Dict[int, List[int]]]} -- mapping from individual series to 
            columns that represent other series that aggregate to this individual series, necessary
            if self_supervised_penalty > 0 (default: {None})
        ignore_future_targets {bool} -- whether to include future targets in forecasting loss
            and past targets in self-supervised reconciliation penalty (default: {False})
        print_rec_penalty {bool} -- whether to print the reconciliation penalty at each step
            of every epoch (default: {True})
        tb_log_dir {bool} -- filepath to which to write tensorboard loss data (default: {False})

    Returns:
        Tuple[GluonPredictor, HybridBlock] -- [description]
    """
    
    if self_supervised_penalty > 0 and hierarchy_agg_dict is None:
        raise ValueError("Must supply 'hierarchy_agg_dict' argument if 'self_supervised_penalty' > 0")

    if embedding_dist_metric != 'cosine' and embedding_dist_metric != 'l2':
        raise ValueError("Embedding distance metric must be either 'cosine' or 'l2'")

    if self_supervised_penalty > 0:
        batch_size = len(training_data)

    if use_cat_var is False:
        cardinality = None

    # set random seeds for reproducibility
    mx.random.seed(0)
    np.random.seed(0)

    estimator = DeepARRecPenaltyEstimator(
        freq=freq, 
        prediction_length=pred_length,
        use_feat_static_cat=use_cat_var,
        cardinality=cardinality,
        embedding_dimension=[int(hidden_dim*embedding_dim_ratio)],
        distr_output=NegativeBinomialOutput(),
        trainer=Trainer(
            epochs=epochs,
            batch_size=batch_size,
            hybridize=False,
            tb_log_dir=tb_log_dir
        ),
        num_layers=num_layers,
        num_cells=hidden_dim,
        self_supervised_penalty=self_supervised_penalty,
        embedding_agg_penalty=embedding_agg_penalty,
        embedding_dist_metric=embedding_dist_metric,
        hierarchy_agg_dict=hierarchy_agg_dict,
        ignore_future_targets=ignore_future_targets,
        print_rec_penalty=print_rec_penalty,
    )

    _, trained_net, predictor = estimator.train_model(training_data = training_data)

    return predictor, trained_net

def fit_predict_arima(
    training_data: ListDataset,
    horizon: int = 12,
    output_file: str = None,
    output_residuals: bool = True,
) -> pd.DataFrame:
    """ for each time series in the training_data individually:
            1) automatically discovers the optimal order for a seasonal ARIMA model 
            2) fits discovered model
            3) makes predictions horizon length into the future

        optionally writes predictions/in-sample residuals to output file

    Arguments:
        training_data {ListDataset} -- training data
    
    Keyword Arugments:
        horizon {int} -- prediction length (default: {12})
        output_file {str} -- output_file to save predictions (default: {None})
        output_residuals {bool} -- whether to output the residuals of in-sample predictions. If True, 
            the in-sample residuals will be prepended to the out-of-sample predictions. Thus, 
            if the in-sample data contains 24 timeteps, and the out-of-sample data contains 6 timesteps,
            the output data frame will contain 30 rows (timesteps) (default: {True})

    Returns:
        pd.DataFrame -- dataframe of point predictions from individually fitted ARIMA models,
            each column represents a series and each row a future point in time
    """

    fits = [
        pm.auto_arima(to_pandas(train_series), suppress_warnings=True, error_action='ignore')
        for train_series in list(training_data)
    ]

    preds = pd.DataFrame([fit.predict(n_periods=horizon) for fit in fits]).T   

    if output_file:
        if output_residuals:
            residuals = pd.DataFrame([
                fit.predict_in_sample() - series['target'] for fit, series in zip(fits, training_data)
            ]).T
            preds = pd.concat([residuals, preds])
        preds.to_csv(output_file, index = False)

    return preds   

def serialize_all(
    fit_models: Tuple[GluonPredictor, HybridBlock],
    base_path: str,
) -> None:
    """ serialize predictors from tuple of fitted models
    
    Arguments:
        fit_models {Tuple[GluonPredictor, HybridBlock]} -- tuples of fitted model objects
        base_path {str} -- base dir name to save
        
    """
    
    for i, (predictor, _) in enumerate(fit_models):
        path = base_path + f"-fold-{i}"
        if not os.path.isdir(path):
            os.mkdir(path)
        predictor.serialize(Path(path))

def unserialize_all(
    base_path: str,
) -> List[GluonPredictor]:
    """ unserialize all predictors from directory
    
    Arguments:
        base_path {str} -- base dir name from which to load models

    Returns:
        List[GluonPredictor] -- list of unserialized GluonPredictor objects
    """
    
    return [
        GluonPredictor.deserialize(Path(base_dir)) 
        for base_dir in glob(base_path + "-fold*")
    ]