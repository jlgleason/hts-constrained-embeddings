# Standard library imports
from typing import List, Optional, Dict, Iterable, Union
import numpy as np
import pandas as pd

# Third-party imports
from gluonts.gluonts_tqdm import tqdm
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import GluonPredictor
from gluonts.evaluation import Evaluator

from .data import LEVEL_NAMES

def evaluate_deepar(
    predictor: GluonPredictor, 
    train_data: ListDataset, 
    test_data: ListDataset, 
    hierarchy_dict: Dict[int, List[int]],
    output_file: str = None,
    output_mean: bool = True,
    output_residuals: bool = True,
) -> Dict[Union[int, str], Dict[str, float]]:
    """ aggregates error metrics for each level of the hierarchy, optionally writes predictions/in-sample residuals 
        to output file
    
    Arguments:
        predictor {GluonPredictor} -- predictor
        train_data {ListDataset} -- train dataset
        test_data {ListDataset} -- test dataset
        hierarchy_dict {Dict[int, List[int]]} -- mapping from hierachy level to series prediction idxs included
            in that level of hierarchy
    
    Keyword Arguments:
        output_file {str} -- output_file to save predictions (default: {None})
        output_mean {bool} -- whether to output the mean (or median) predictions (default: {False})
        output_residuals {bool} -- whether to output the residuals of in-sample predictions. If True, 
            the in-sample residuals will be prepended to the out-of-sample predictions. Thus, 
            if the in-sample data contains 24 timeteps, and the out-of-sample data contains 6 timesteps,
            the output data frame will contain 30 rows (timesteps) (default: {True})

    Returns:
        Dict[Union[int, str], Dict[str, float]] -- mapping of hierarchy level (0-indexed) to dictionaries of aggregated metrics 
            for that level of the hierarchy
    """

    eval_forecasts = []
    output_forecasts = []
    with tqdm(
        predictor.predict(train_data),
        total=len(train_data),
        desc="Making Predictions"
    ) as it, np.errstate(invalid='ignore'):
        for forecast in it:
            output_forecasts.append(forecast.mean if output_mean else forecast.quantile(0.5))
            eval_forecasts.append(forecast)
        preds = np.array(output_forecasts)

    
    if output_file:
        if output_residuals:
            preds = np.concatenate(
                (predictor.prediction_net.residuals.asnumpy(), preds), 
                axis=1
            )
        out_df = pd.DataFrame(preds).T
        out_df.to_csv(output_file, index = False)

    eval_forecasts = np.array(eval_forecasts)
    evaluator = Evaluator(quantiles=[0.5])
    evaluations = {
        level: evaluator(
            [to_pandas(series) for series in np.array(list(test_data))[np.array(idxs)]],
            eval_forecasts[np.array(idxs)]
        )[0]
        for level, idxs in hierarchy_dict.items()
    }
    evaluations['all'] = evaluator(
        [to_pandas(series) for series in np.array(list(test_data))],
        eval_forecasts
    )[0]
    return evaluations

class PointEstimateEvaluator(Evaluator):
    """ PointEstimateEvaluator object that evaluates point estimates instead of probabilistic forecasts
    """

    def __init__(
        self,
    ) -> None:
        pass

    def __call__(
        self,
        ts_iterator: Iterable[Union[pd.DataFrame, pd.Series]],
        point_estimates: Iterable[Union[np.array, pd.Series]],
        num_series: Optional[int] = None,
    ) -> Dict[str, float]:
        """ Compute accuracy metrics by comparing actual data to the point estimates.
        
        Arguments:
            ts_iterator {Iterable[Union[pd.DataFrame, pd.Series]]} -- iterator containing true target on the predicted range
            point_estimates {np.array} -- iterator containing point estimates on the predicted range
        
        Keyword Arguments:
            num_series {Optional[int]} -- number of series of the iterator (optional, only used for displaying progress)
                (default: {None})
        
        Returns:
            Dict[str, float -- Dictionary of aggregated metrics
        """
        ts_iterator = iter(ts_iterator)
        fcst_iterator = iter(point_estimates)
        
        rows = []

        with tqdm(
            zip(ts_iterator, fcst_iterator),
            total=num_series,
            desc="Running evaluation",
        ) as it, np.errstate(invalid="ignore"):
            for ts, forecast in it:
                rows.append(self.get_metrics_per_ts(ts, forecast))

        assert not any(
            True for _ in ts_iterator
        ), "ts_iterator has more elements than fcst_iterator"

        assert not any(
            True for _ in fcst_iterator
        ), "fcst_iterator has more elements than ts_iterator"

        if num_series is not None:
            assert (
                len(rows) == num_series
            ), f"num_series={num_series} did not match number of elements={len(rows)}"

        # If all entries of a target array are NaNs, the resulting metric will have value "masked". Pandas does not
        # handle masked values correctly. Thus we set dtype=np.float64 to convert masked values back to NaNs which
        # are handled correctly by pandas Dataframes during aggregation.
        metrics_per_ts = pd.DataFrame(rows, dtype=np.float64)
        return self.get_aggregate_metrics(metrics_per_ts)

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], point_estimates: Union[np.array, pd.Series]
    ) -> Dict[str, Union[float, str, None]]:
        """ get metrics on predictions for an individual time series
        
        Arguments:
            time_series {Union[pd.Series, pd.DataFrame]} -- true targets on the predicted range
            point_estimates {Union[np.array, pd.Series]} -- point estimates on the predicted range
        
        Returns:
            Dict[str, Union[float, str, None]] -- Dictionary of individual metrics
        """

        pred_target = np.ma.masked_invalid(time_series.values)

        metrics = {
            "MSE": self.mse(pred_target, point_estimates),
            "abs_error": self.abs_error(pred_target, point_estimates),
            "abs_target_sum": self.abs_target_sum(pred_target),
            "abs_target_mean": self.abs_target_mean(pred_target),
            "MAPE": self.mape(pred_target, point_estimates),
        }

        return metrics

    def get_aggregate_metrics(
        self, metric_per_ts: pd.DataFrame
    ) -> Dict[str, float]:
        """ aggregate metrics from evaluations of individual time series
        
        Arguments:
            metric_per_ts {pd.DataFrame} -- data frame of metrics for each individual time series  
        
        Returns:
            Dict[str, float] -- Dictionary of aggregate metrics
        """
        agg_funs = {
            "MSE": "mean",
            "abs_error": "sum",
            "abs_target_sum": "sum",
            "abs_target_mean": "mean",
            "MAPE": "mean",
        }
        assert (
            set(metric_per_ts.columns) >= agg_funs.keys()
        ), "The some of the requested item metrics are missing."

        totals = {
            key: metric_per_ts[key].agg(agg) for key, agg in agg_funs.items()
        }

        # derived metrics based on previous aggregate metrics
        totals["RMSE"] = np.sqrt(totals["MSE"])

        flag = totals["abs_target_mean"] == 0
        totals["NRMSE"] = np.divide(
            totals["RMSE"] * (1 - flag), totals["abs_target_mean"] + flag
        )

        flag = totals["abs_target_sum"] == 0
        totals["ND"] = np.divide(
            totals["abs_error"] * (1 - flag), totals["abs_target_sum"] + flag
        )

        return totals

def evaluate_optimal_rec(
    predictions: pd.DataFrame, 
    test_data: ListDataset, 
    hierarchy_dict: Dict[int, List[int]],
) -> Dict[str, Dict[str, float]]:
    """ aggregates error metrics for each level of the hierarchy, calculated over data frame of point
        estimates (for example, those returned after optimal reconciliation) instead of probabilistic 
        forecast objects

    Arguments:
        predictions {pd.DataFrame} -- data frame of point predictions
        test_data {ListDataset} -- test dataset
        hierarchy_dict {Dict[int, List[int]]} -- mapping from hierachy level to series prediction idxs included
            in that level of hierarchy

    Returns:
        Dict[str, Dict[str, float]] -- mapping of hierarchy level (0-indexed) to dictionaries of aggregated metrics 
            for that level of the hierarchy
    """

    evaluator = PointEstimateEvaluator()
    evaluations = {
        level: evaluator(
            [to_pandas(series) for series in np.array(list(test_data))[np.array(idxs)]], 
            predictions.values.T[np.array(idxs)],
        )
        for level, idxs in hierarchy_dict.items()
    }
    evaluations['all'] = evaluator(
            [to_pandas(series) for series in np.array(list(test_data))], 
            predictions.values.T,
        )
    return evaluations

def agg_evaluations(
    evaluations: List[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, List[float]]]:
    """ aggregates error metrics from models fit on different CV folds
    
    Arguments:
        evaluations {List[Dict[str, Dict[str, float]]]} -- list of evaluation metrics
            on individual CV folds, one for each fold
    
    Returns:
        Dict[str, Dict[str, List[float]]] -- mapping of hierarchy level (0-indexed) to dictionaries of aggregated metrics
            (mean and std) over all folds for that level of the hierarchy
    """
    
    cumulative_evals = {level: {
            metric: [] for metric, _ in evaluations[0]['all'].items()
        } 
        for level, _ in evaluations[0].items()
    }

    for individual_eval in evaluations:
        for level, level_eval in individual_eval.items():
            for metric, metric_eval in level_eval.items():
                cumulative_evals[level][metric].append(metric_eval)
    
    for level, agg_eval in cumulative_evals.items():
        for metric, metric_stats in agg_eval.items():
            mean = np.mean(metric_stats)
            std = np.std(metric_stats)
            cumulative_evals[level][metric] = [mean, std]

    return cumulative_evals

DEFAULT_MODEL_NAMES = ['Baseline Arima', 'Baseline DeepAR', 'DeepAR-Cat-Var', 'DeepAR-MinT-Rec', 'DeepAR-Self-Supervised', 'DeepAR-Embedding-Agg']

def compare_performance(
    agg_evaluations: List[Dict[Union[int, str], Dict[str, float]]],
    model_names: List[str] = DEFAULT_MODEL_NAMES,
    metrics: List[str] = ['RMSE', 'NRMSE', 'ND'],
    levels: List[str] = LEVEL_NAMES,
    outfile: str = None,
) -> None:
    """ compares the performance of an arbitrary number of models (aggregated over all CV folds)
            on any of the metrics collected during the individual evaluations

    Arguments:
        agg_evaluations {List[Dict[Union[int, str], Dict[str, float]]]} -- list of aggregated
            performance metrics for an arbitrary number of models
    
    Keyword Arguments:
        model_names {List[str]} -- name of model that is represented in each index of agg_evaluations, used
            for printing (default: {default_model_names})
        metrics {List[str]} -- list of metrics to assess comparison on 
            (default: {['RMSE', 'NRMSE', 'MAPE', 'ND']})
        levels {List[str]} -- levels of the hierarchy that we want to compare performance
            over (default: {LEVEL_NAMES}))
        outfile {str} -- filepath to which to write formatted comparison metrics (default: {None}))
    """

    if len(agg_evaluations) != len(model_names):
        raise ValueError("The number of aggregate evaluations and the number of model names must " + 
            "have the same length")
    
    if not all([metric in agg_evaluations[0]['all'].keys() for metric in metrics]):
        raise ValueError("All metrics must be included in all aggregate evaluations")

    if outfile:
        f = open(outfile, "w+")
    
    def write_output(print_string):
        print(print_string)
        if outfile:
            f.write(f'{print_string}\n')

    for metric in metrics:
        write_output(f"-----------------------------------------------------")
        write_output(f"{'MODEL':21}{metric} {'(mean +- std)':17} % Change")
        write_output(f"-----------------------------------------------------\n")
        for level in levels:
            write_output(f'{level}:')
            sort_performance = sorted(
                list(zip(
                    model_names, 
                    [e[level][metric][0] for e in agg_evaluations], # means
                    [e[level][metric][1] for e in agg_evaluations], # stds
                )),
                key = lambda x: x[1]
            )
            baseline = agg_evaluations[0][level][metric][0]
            for model_name, mean, std in sort_performance:
                write_output(f"{model_name:20} {mean.round(3):8} +- {std.round(3):7} " +
                      f"{round((baseline - mean) / baseline * 100, 3):10}%")
            write_output('')
    
    if outfile:
        f.close()



