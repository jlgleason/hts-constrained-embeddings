# Standard library imports
from typing import Dict, List, Optional, Iterator
import numpy as np

# Third-party imports
from mxnet.gluon import HybridBlock
from gluonts.core.component import validated
from gluonts.support.util import weighted_average, copy_parameters
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import Dataset
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
    ExpectedNumInstanceSampler
)
from gluonts.distribution.distribution import getF
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepar._network import DeepARTrainingNetwork, DeepARPredictionNetwork
from gluonts.model.common import Tensor
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.model.forecast import Forecast

class DeepARRecPenaltyTrainingNetwork(DeepARTrainingNetwork):
    """ Construct a DeepARTrainingNetwork that adds a self-supervised and/or embedding 
        reconciliation penalty to its likelihood loss.

        If ignore_future_targets is True, we only calculate forecasting loss on past targets (training set)
        and reconciliation loss on future targets as a more accurate approximation of the 
        self-supervised reconciliation penalty in "A Self-supervised Approach to Hierarchical Forecasting 
        with Applications to Groupwise Synthetic Controls"
    
        Self-supervised reconciliation penalty:

            t_0 < t < T := time steps (if t_0 == 0 we consider all time steps)
            c := total number of aggregation constraints
            j_0 := index of aggregate series in constraint j
            j_1 ... j_m := indices of disaggregated series in constraint j that sum to j_0 series

            lambda * sum_{t_0 < t < T, 0 <= j < c}(
                (Y_hat_j_0 - sum_{j_1 <= j_k <= j_m}(Y_hat_j_k))^2
            )
        
        Embedding reconciliation penalty:
            
            t_0 < t < T := time steps (if t_0 == 0 we consider all time steps)
            c := total number of aggregation constraints
            j_0 := index of aggregate series in constraint j
            j_1 ... j_m := indices of disaggregated series in constraint j that sum to j_0 series
            X_cat_j_k := 1-hot categorical value of series at index k in constraint j
            E(X_cat_j_k) := learned embedding of this previous value

            lambda * sum_{t_0 < t < T, 0 <= j < c}(
                l2_norm(
                    E(X_cat_j_0) - sum_{j_1 <= j_k <= j_m}E(X_cat_j_k)
                )^2
            )

            or 

            lambda * sum_{t_0 < t < T, 0 <= j < c}(
                cosine_distance(
                    E(X_cat_j_0) - sum_{j_1 <= j_k <= j_m}E(X_cat_j_k)
                )^2
            )
    """
    
    def __init__(
        self,
        *args,
        self_supervised_penalty: float = 0,
        embedding_agg_penalty: float = 0,
        embedding_dist_metric: str = 'l2',
        hierarchy_agg_dict: Dict[int, List[int]] = None,
        ignore_future_targets: bool = False,
        print_rec_penalty: bool = True,
        **kwargs,
    ) -> None:
        """ 

        Keyword Arguments:
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
            print_rec_penalty {bool} -- whether to print the reconciliation penalty at each step
                of every epoch (default: {True})

        """

        super().__init__(*args, **kwargs)

        if self_supervised_penalty > 0 and hierarchy_agg_dict is None:
            raise ValueError("Must supply 'hierarchy_agg_dict' argument if 'self_supervised_penalty' > 0")

        if embedding_dist_metric != 'cosine' and embedding_dist_metric != 'l2':
            raise ValueError("Embedding distance metric must be either 'cosine' or 'l2'")
        
        self.self_supervised_penalty = self_supervised_penalty
        self.embedding_agg_penalty = embedding_agg_penalty
        self.embedding_dist_metric = embedding_dist_metric
        self.hierarchy_agg_dict = hierarchy_agg_dict
        self.ignore_future_targets = ignore_future_targets
        self.print_rec_penalty = print_rec_penalty
    
    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,
        feat_static_real: Tensor,
        past_time_feat: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        future_time_feat: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:

        if self.ignore_future_targets:
            
            distr = self.distribution(
                feat_static_cat=feat_static_cat,
                feat_static_real=feat_static_real,
                past_time_feat=past_time_feat,
                past_target=past_target,
                past_observed_values=past_observed_values,
                future_time_feat=None,
                future_target=None,
                future_observed_values=future_observed_values,
            )
            
            loss = distr.loss(
                past_target.slice_axis(
                    axis=1,
                    begin=self.history_length - self.context_length,
                    end=None,
                )
            )
            
            # (batch_size, seq_len, *target_shape)
            observed_values = past_observed_values.slice_axis(
                axis=1,
                begin=self.history_length - self.context_length,
                end=self.history_length,
            )
            
            # mask the loss at one time step iff one or more observations is missing in the target dimensions
            # (batch_size, seq_len)
            loss_weights = (
                observed_values
                if (len(self.target_shape) == 0)
                else observed_values.min(axis=-1, keepdims=False)
            )

            weighted_loss = weighted_average(
                F=F, x=loss, weights=loss_weights, axis=1
            )

        else:
            
            distr = self.distribution(
                feat_static_cat=feat_static_cat,
                feat_static_real=feat_static_real,
                past_time_feat=past_time_feat,
                past_target=past_target,
                past_observed_values=past_observed_values,
                future_time_feat=future_time_feat,
                future_target=future_target,
                future_observed_values=future_observed_values,
            )
            
            # put together target sequence
            # (batch_size, seq_len, *target_shape)
            target = F.concat(
                past_target.slice_axis(
                    axis=1,
                    begin=self.history_length - self.context_length,
                    end=None,
                ),
                future_target,
                dim=1,
            )

            # (batch_size, seq_len)
            loss = distr.loss(target)

            # (batch_size, seq_len, *target_shape)
            observed_values = F.concat(
                past_observed_values.slice_axis(
                    axis=1,
                    begin=self.history_length - self.context_length,
                    end=self.history_length,
                ),
                future_observed_values,
                dim=1,
            )

            # mask the loss at one time step iff one or more observations is missing in the target dimensions
            # (batch_size, seq_len)
            loss_weights = (
                observed_values
                if (len(self.target_shape) == 0)
                else observed_values.min(axis=-1, keepdims=False)
            )

            weighted_loss = weighted_average(
                F=F, x=loss, weights=loss_weights, axis=1
            )
        
        total_loss = F.sum(weighted_loss) / weighted_loss.shape[0]
        print_string = f'Forecasting loss: {total_loss.asscalar()}'

        f_loss = F.array([0.0])
        e_loss = F.array([0.0])
        for _, (agg_idx, disagg_idxs) in self.hierarchy_agg_dict.items():
            
            # add self-supervised reconciliation loss
            if self.self_supervised_penalty > 0:
                agg_preds = F.reshape(
                    F.take(distr.mean, F.array([agg_idx])),
                    shape = (-1,)
                )
                disagg_preds = F.take(distr.mean, F.array(disagg_idxs))
                f_loss_i = F.sum(F.square(agg_preds - F.sum(disagg_preds, axis=0)))
                f_loss = F.concat(f_loss, f_loss_i, dim=0)
        
            # add embedding reconciliation loss
            if self.embedding_agg_penalty > 0:
                embedded = self.embedder(
                    F.expand_dims(F.array([i for i in range(self.cardinality[0])]), axis=1)
                )
                agg_embed = F.take(embedded, F.array([agg_idx]))
                disagg_embed = F.sum(F.take(embedded, F.array(disagg_idxs)), axis = 0)

                if self.embedding_dist_metric == 'cosine':
                    e_loss_i = 1 - (F.dot(agg_embed, disagg_embed) / F.norm(agg_embed) / F.norm(disagg_embed))
                else:
                    e_loss_i = F.square(F.norm(agg_embed - disagg_embed))
                e_loss = F.concat(e_loss, e_loss_i, dim=0)

        if self.self_supervised_penalty > 0:
            total_f_loss = F.sum(f_loss) / weighted_loss.shape[0] / len(self.hierarchy_agg_dict)
            total_loss = total_loss + total_f_loss * F.array([self.self_supervised_penalty])
        
        if self.embedding_agg_penalty > 0:
            total_e_loss = F.sum(e_loss) / weighted_loss.shape[0] / len(self.hierarchy_agg_dict)
            total_loss = total_loss + total_e_loss * F.array([self.embedding_agg_penalty])

        # print forecasting/reconciliation loss at each step
        if self.print_rec_penalty:

            if self.self_supervised_penalty > 0:
                print_string = print_string + f', Self-supervised Loss: {total_f_loss.asscalar()}'
            
            if self.embedding_agg_penalty > 0:
                print_string = print_string + f', Embedding agg Loss: {total_e_loss.asscalar()}'
            
            print(print_string)        

        return total_loss, loss

class DeepARRecPenaltyPredictionNetwork(DeepARPredictionNetwork):
    """ Construct a DeepARPredictionNetwork that can store in-sample predictions (needed for evaluating
            MinT reconciliation approach)
    """
    
    def __init__(
        self,
        *args,
        store_in_sample_residuals: bool = True,
        **kwargs,
    ) -> None:
        """ 

        Keyword Arguments:
            store_in_sample_residuals {bool} -- whether to store in-sample train predictions when forecasting
                test data (default: {True})
        """
        super().__init__(*args, **kwargs)
        self.store_in_sample_residuals = store_in_sample_residuals
        self.residuals = None
        self.residuals_complete = False

    def in_sample_residuals(
        self,
        rnn_outputs: Tensor,
        scale: Tensor,
        target: Tensor,
        F
    ) -> Tensor:
        """ calculates the residuals of in-sample predictions 
            (difference between distribution mean and target value)
        
        Arguments:
            rnn_outputs {Tensor} -- output values of RNN at each in-sample timestep
            scale {Tensor} -- scale parameter needed for specific distributions
            past_targets {Tensor} -- target values from training set
        
        Returns:
            Tensor -- in-sample residuals
        """
        
        distr_args = self.proj_distr_args(rnn_outputs)
        distr = self.distr_output.distribution(distr_args, scale=scale)
        target = target.slice_axis(
            axis=1,
            begin=self.history_length - self.context_length,
            end=None,
        )
        residual = distr.mean - target

        if self.residuals is None:
            self.residuals = residual
        else:
            self.residuals = F.concat(self.residuals, residual, dim = 0)

    def hybrid_forward(
        self,
        F,
        feat_static_cat: Tensor,  # (batch_size, num_features)
        feat_static_real: Tensor,  # (batch_size, num_features)
        past_time_feat: Tensor,  # (batch_size, history_length, num_features)
        past_target: Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: Tensor,  # (batch_size, prediction_length, num_features)
    ) -> Tensor:

        # unroll the decoder in "prediction mode", i.e. with past data only
        rnn_outputs, state, scale, static_feat = self.unroll_encoder(
            F=F,
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=None,
            future_target=None,
        )

        if self.store_in_sample_residuals:
            if not self.residuals_complete:
                self.in_sample_residuals(
                    rnn_outputs, 
                    scale,
                    past_target,
                    F
                )

        return self.sampling_decoder(
            F=F,
            past_target=past_target,
            time_feat=future_time_feat,
            static_feat=static_feat,
            scale=scale,
            begin_states=state,
        )

class RepresentableBlockPredictorResiduals(RepresentableBlockPredictor):
    """ construct a RepresentableBlockPredictor object that will calculate the in-sample
        prediction residuals once and only once
    """

    def predict(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        **kwargs,
    ) -> Iterator[Forecast]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            ctx=self.ctx,
            dtype=self.dtype,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            **kwargs,
        )
        yield from self.forecast_generator(
            inference_data_loader=inference_data_loader,
            prediction_net=self.prediction_net,
            input_names=self.input_names,
            freq=self.freq,
            output_transform=self.output_transform,
            num_samples=num_samples,
        )

        self.prediction_net.residuals_complete = True

class FixedUnitSampler(InstanceSampler):
    """
    Samples exactly one window from each time series
    """

    @validated()
    def __init__(self) -> None:
        pass

    def __call__(self, ts: np.ndarray, a: int, b: int) -> np.ndarray:
        assert (
            a <= b
        ), "First index must be less than or equal to the last index."
        
        return (np.random.randint(a,b+1),)



class DeepARRecPenaltyEstimator(DeepAREstimator):
    """ Construct a DeepAREstimator that adds a self-supervised and/or embedding 
        reconciliation penalty to its likelihood loss.

        We overwrite the create_transformation() method to use a FixedUnitSampler() in the 
        InstanceSplitter() step, which samples exactly one window from each training series. 
        In each batch, it is necessary to have one sample of each individual series from the 
        training set to calculate the self-supervised reconciliation penalty, which is applied
        over all samples from the training set. This is necessary because the DeepAREstimator learns
        a univariate global model for the dynamics and thus the randomized block coordinate descent
        optimization algorithm proposed in "A Self-supervised Approach to Hierarchical Forecasting 
        with Applications to Groupwise Synthetic Controls" is not applicable. 
    """
    
    def __init__(
        self, 
        *args,
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

        if self_supervised_penalty > 0:
            if hierarchy_agg_dict is None:
                raise ValueError("Must supply 'hierarchy_agg_dict' argument if 'self_supervised_penalty' > 0")
        self.train_sampler = FixedUnitSampler()
        #else:
        #    self.train_sampler = ExpectedNumInstanceSampler(num_instances=1)
        if embedding_dist_metric != 'cosine' and embedding_dist_metric != 'l2':
            raise ValueError("Embedding distance metric must be either 'cosine' or 'l2'")

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
                    train_sampler=self.train_sampler,
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

    

