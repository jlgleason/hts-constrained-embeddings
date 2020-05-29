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
from gluonts.transform.sampler import InstanceSampler, BucketInstanceSampler
from gluonts.trainer import Trainer
from gluonts.distribution import NegativeBinomialOutput, GaussianOutput
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.predictor import Predictor, GluonPredictor
from gluonts.model.deepar import DeepAREstimator
from gluonts.support.util import copy_parameters
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    InstanceSplitter,
    RemoveFields,
    SetField,
    Transformation,
    VstackFeatures,
    InstanceSampler,
)
import pmdarima as pm

# First Party imports
from .data import FixedUnitSampler
from .network import (
    DeepARRecPenaltyTrainingNetwork, 
    DeepARRecPenaltyPredictionNetwork,
    RepresentableBlockPredictorResiduals
)

class DeepARRecPenaltyEstimator(DeepAREstimator):
    """ Construct a DeepAREstimator that optionally adds a self-supervised and/or embedding 
        reconciliation penalty to its likelihood loss.

        We overwrite the create_transformation() method to use a different InstanceSampler object in the
        InstanceSplitter step. Specifically, if the self-supervised loss is included, we use a FixedUnitSampler, 
        which samples exactly one window from each training series. This is because, in each batch, it is 
        necessary to have one sample of each individual series from the  training set to calculate the 
        self-supervised reconciliation penalty, which is applied over all samples from the training set.
        If the self-supervised loss is not included, we use a BucketInstanceSampler, in whcih the 
        probability of sampling from bucket i is the inverse of its number of elements

    """
    
    def __init__(
        self, 
        *args,
        sampler: InstanceSampler = FixedUnitSampler(),
        self_supervised_penalty: float = 0,
        embedding_agg_penalty: float = 0,
        embedding_dist_metric: str = 'cosine',
        hierarchy_agg_dict: Dict[int, List[int]] = None,
        ignore_future_targets: bool = False,
        store_in_sample_residuals: bool = True,
        print_rec_penalty: bool = True,
        **kwargs
    ) -> None:
        """
        
        Keyword Arguments:
            sampler {InstanceSampler} -- GluonTS sampler object containing logic for how to sample mini-batches
                (default: {FixedUnitSampler()})
            self_supervised_penalty {float} -- lambda for self-supervised reconciliation penalty 
                (default: {0.0})
            embedding_agg_penalty {float} -- lambda for embedding rec. penalty 
                (default: {0.0})
            embedding_dist_metric {str} -- distance metric for embedding rec. penalty
                (default: {'cosine'})
            hierarchy_agg_dict {Optional[Dict[int, List[int]]]} -- mapping from individual series to 
                columns that represent other series that aggregate to this individual series, necessary
                if self_supervised_penalty > 0 (default: {None})
            ignore_future_targets {bool} -- whether to include future targets in forecasting loss
                and past targets in self-supervised reconciliation penalty (default: {False})
            store_in_sample_residuals {bool} -- whether to store in-sample train predictions when forecasting
                test data (default: {True})
            print_rec_penalty {bool} -- whether to print the reconciliation penalty at each step
                of every epoch (default: {True})

        """
        super().__init__(*args, **kwargs)

        if self_supervised_penalty > 0 and hierarchy_agg_dict is None:
            raise ValueError("Must supply 'hierarchy_agg_dict' argument if 'self_supervised_penalty' > 0")

        if embedding_agg_penalty > 0 and hierarchy_agg_dict is None:
            raise ValueError("Must supply 'hierarchy_agg_dict' argument if 'embedding_agg_penalty' > 0")

        if embedding_dist_metric != 'cosine' and embedding_dist_metric != 'l2':
            raise ValueError("Embedding distance metric must be either 'cosine' or 'l2'")

        self.sampler = sampler
        self.self_supervised_penalty = self_supervised_penalty
        self.hierarchy_agg_dict = hierarchy_agg_dict
        self.ignore_future_targets = ignore_future_targets
        self.print_rec_penalty = print_rec_penalty
        self.store_in_sample_residuals = store_in_sample_residuals
        self.embedding_agg_penalty = embedding_agg_penalty
        self.embedding_dist_metric = embedding_dist_metric
        
    def create_transformation(self) -> Transformation:
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if not self.use_feat_static_real:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0])]
                if not self.use_feat_static_cat
                else []
            )
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                    )
                ]
                if not self.use_feat_static_real
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # in the following line, we add 1 for the time dimension
                    expected_ndim=1 + len(self.distr_output.event_shape),
                    dtype=self.dtype,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dummy_value=self.distr_output.value_in_support,
                    dtype=self.dtype,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                    dtype=self.dtype,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    ),
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=self.sampler,
                    past_length=self.history_length,
                    future_length=self.prediction_length,
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.OBSERVED_VALUES,
                    ],
                    dummy_value=self.distr_output.value_in_support,
                ),
            ]
        )
    
    def create_training_network(self) -> DeepARRecPenaltyTrainingNetwork:
        return DeepARRecPenaltyTrainingNetwork(
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            dtype=self.dtype,
            self_supervised_penalty=self.self_supervised_penalty,
            hierarchy_agg_dict=self.hierarchy_agg_dict,
            ignore_future_targets=self.ignore_future_targets,
            print_rec_penalty=self.print_rec_penalty,
            embedding_agg_penalty=self.embedding_agg_penalty,
            embedding_dist_metric=self.embedding_dist_metric,
        )
    
    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = DeepARRecPenaltyPredictionNetwork(
            num_parallel_samples=self.num_parallel_samples,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            dtype=self.dtype,
            store_in_sample_residuals=self.store_in_sample_residuals,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictorResiduals(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            dtype=self.dtype,
        )


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
    sampler: Optional[InstanceSampler] = None,
    self_supervised_penalty: float = 0.0,
    embedding_agg_penalty: float = 0,
    embedding_dist_metric: str = 'cosine',
    embedding_dim_ratio: float = 1,
    hierarchy_agg_dict: Optional[Dict[int, List[int]]] = None,
    ignore_future_targets: bool = False,
    print_rec_penalty: bool = True,
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
        sampler {Optional[InstanceSampler]} -- GluonTS sampler object containing logic for how to sample mini-batches
            (default: {None})
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

    Returns:
        Tuple[GluonPredictor, HybridBlock] -- [description]
    """
    
    if self_supervised_penalty > 0 and hierarchy_agg_dict is None:
        raise ValueError("Must supply 'hierarchy_agg_dict' argument if 'self_supervised_penalty' > 0")

    if embedding_agg_penalty > 0 and hierarchy_agg_dict is None:
        raise ValueError("Must supply 'hierarchy_agg_dict' argument if 'embedding_agg_penalty' > 0")

    if embedding_dist_metric != 'cosine' and embedding_dist_metric != 'l2':
        raise ValueError("Embedding distance metric must be either 'cosine' or 'l2'")

    if self_supervised_penalty > 0:
        batch_size = len(training_data)
        sampler = FixedUnitSampler()
    else:
        sampler = sampler

    if use_cat_var is False:
        cardinality = None

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
        ),
        num_layers=num_layers,
        num_cells=hidden_dim,
        sampler=sampler,
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