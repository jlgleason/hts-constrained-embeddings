# Standard library imports
from typing import Tuple, List, Dict, Iterable
import re
import pandas as pd
import numpy as np

# Third Party imports
from sklearn.model_selection import TimeSeriesSplit
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset

TRAVEL_TYPES = 4

LEVEL_NAMES = [
    'country',
    'state',
    'zone',
    'region', 
    'country-by-travel', 
    'state-by-travel', 
    'zone-by-travel', 
    'region-by-travel'
]

def unique_prefixes(
    strings: Iterable,
    prefix_lengths: List[int]
) -> List[List[str]]:
    """ find unique prefixes in Iterable, for each length in `prefix_lengths`
    
    Arguments:
        strings {Iterable} -- iterable to search over
        prefix_lengths {List[int]} -- list of lengths of prefixes
    
    Returns:
        List[List[str]] -- list of lists of unique prefixes (one for each length in
            prefix_lengths)
    """

    prefixes = [
        list(set([string[:prefix_length] for string in strings]))
        for prefix_length in prefix_lengths
    ]
    [prefix_list.sort() for prefix_list in prefixes]
    return prefixes

def regex_labels(
    travel_types: List[str],
    prefixes: List[List[str]]
) -> List[List[re.Pattern]]:
    """ generate lists of regexs (one list for each level of hierarchy) from
        types of travel and location prefixes

    Arguments:
        travel_types {List[str]} -- types of travel
        prefixes {List[List[str]]} -- location prefixes (one list for each level of hierarchy)

    Returns:
        List[List[re.Pattern]] -- location regexes (one list for each level of hierarchy)
    """
    
    prefix_1, prefix_2, prefix_3 = prefixes

    region_labels = [
        re.compile(f'{region}.+') 
        for region in prefix_3
    ]
    zone_labels = [
        re.compile(f'{zone}.+') 
        for zone in prefix_2
    ]
    state_labels = [
        re.compile(f'{state}.+') 
        for state in prefix_1
    ] 
    zone_by_travel_labels = [
        re.compile(f'{zone}.{travel}')
        for zone in prefix_2
        for travel in travel_types
    ]
    state_by_travel_labels = [
        re.compile(f'{state}..{travel}')
        for state in prefix_1
        for travel in travel_types
    ]
    country_by_travel_labels = [
        re.compile(f'...{travel}') 
        for travel in travel_types
    ]

    return [
        region_labels, 
        zone_labels,
        state_labels, 
        zone_by_travel_labels, 
        state_by_travel_labels,
        country_by_travel_labels
    ]

def sum_filtered_cols(
    df: pd.DataFrame,
    filter_cols: List[re.Pattern],
    sort: bool = True,
) -> pd.DataFrame:
    """ sums df columns that match regular expressions in 'filter_cols' along column 
        (series, not time) axis, sorts columns of resulting df lexicographically
    
    Arguments:
        df {pd.DataFrame} -- data frame
        filter_col_names {List[re.Pattern]} -- roots of strings that should match for summation
        
    Keyword Arguments:
        sort {bool} -- whether to sort columns of df lexicographically before returning 
            (default: {True})
    
    Returns:
        pd.DataFrame -- summed and (optionally) lexicographically sorted data frame
    """

    new_df = pd.DataFrame(index = df.index)
    for regex in filter_cols:
        new_df[regex.pattern] = df[
            list(filter(regex.match, df.columns))
        ].sum(axis=1)
    if sort:
        new_df = new_df.reindex(sorted(new_df.columns), axis = 1)
    return new_df

def all_dfs(
    tourism_df: pd.DataFrame,
    labels: List[List[re.Pattern]]
) -> pd.DataFrame:
    """ generate and combine dataframes for all levels of the hierarchy

    Arguments:
        tourism_df {pd.DataFrame} -- tourism frame with bottom level series (region/type)
        labels {List[List[re.Pattern]]} -- location regexes (one list for each level of hierarchy)

    Returns:
        pd.DataFrame -- single combined dataframe
    """

    region_labels, zone_labels, state_labels, zone_by_travel_labels, state_by_travel_labels, country_by_travel_labels = labels

    region_df = sum_filtered_cols(tourism_df, region_labels)
    zone_df = sum_filtered_cols(tourism_df, zone_labels)
    state_df = sum_filtered_cols(tourism_df, state_labels)
    zone_by_travel_df = sum_filtered_cols(
        tourism_df, 
        zone_by_travel_labels,
        sort=False    
    )
    state_by_travel_df = sum_filtered_cols(
        tourism_df, 
        state_by_travel_labels, 
        sort=False
    )
    country_df = sum_filtered_cols(tourism_df, [re.compile(f'.+')])
    country_by_travel_df = sum_filtered_cols(
        tourism_df, 
        country_by_travel_labels,
        sort=False
    )
    return pd.concat(
        [
            country_df, 
            state_df, 
            zone_df, 
            region_df, 
            country_by_travel_df,
            state_by_travel_df, 
            zone_by_travel_df,
            tourism_df
        ], 
        axis=1
    )

def level_counts(
    prefixes: List[List[str]],
    labels: List[List[re.Pattern]],
) -> List[List[int]]:
    """ generate list of the counts of leaves that sum to each node in the hierarchy

    Arguments:
        prefixes {List[List[str]]} -- location prefixes (one list for each level of hierarchy)
        labels {List[List[re.Pattern]]} -- location regexes (one list for each level of hierarchy)

    Returns:
        List[List[int]] -- list of the counts of leaves that sum to each node, 
            one list for each level in the hierarchy
    """

    prefix_1, prefix_2, prefix_3 = prefixes
    _, zone_labels, state_labels, _, _, _ = labels

    country_cts = [1]
    state_cts = [len(prefix_1)]
    zone_cts = [
        len(list(filter(regex.match, prefix_2))) 
        for regex in state_labels
    ]
    region_cts = [
        len(list(filter(regex.match, prefix_3))) 
        for regex in zone_labels
    ]
    return [country_cts, state_cts, zone_cts, region_cts]

def preprocess(
    tourism_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[int]]:
    """ preprocess hierarchical tourism df into one df with all level series
    
    Arguments:
        tourism_df {pd.DataFrame} -- tourism frame with bottom level series (region/type)
    
    Returns:
        Tuple[pd.DataFrame, List[int]] -- 
            1) new df with region, zone, state, and country series as separate columns
            2) list of the counts of leaves that sum to each node in the hierarchy
    """

    tourism_df['Year'] = tourism_df['Year'].ffill().astype(int)
    tourism_df['Date'] = tourism_df['Month'] + ' ' + tourism_df['Year'].astype(str)
    tourism_df = tourism_df.drop(columns = ['Year', 'Month'])
    tourism_df = tourism_df.set_index('Date')

    travel_types = [col[-3:] for col in tourism_df.columns[:4]]
    prefixes = unique_prefixes(tourism_df.columns, [1,2,3])
    labels = regex_labels(travel_types, prefixes)
    all_df = all_dfs(tourism_df, labels)
    counts = level_counts(prefixes, labels)

    return all_df, counts

def make_hierarchy_level_dict(
    df: pd.DataFrame,
    level_counts: List[List[int]]
) -> Dict[str, List[int]]:
    """ creates dict of which columns (series) belong to which level of hierarchy 
    
    Arguments:
        df {pd.DataFrame} -- data frame with all series
        level_counts {List[List[int]]} -- list of the counts of leaves that sum to each node, 
            one list for each level in the hierarchy
    
    Returns:
        Dict[str, [List[int]]] -- mapping from hierachy level to columns in that level of hierarchy
    """

    def idx_func(idx_list):
        return [i for i in range(len(idx_list))]

    level_agg = level_counts[0] + level_counts[1] + [sum(level_counts[2])] + [sum(level_counts[3])]
    level_agg += [l * TRAVEL_TYPES for l in level_agg]

    return {
        level: [idx for idx in range(sum(level_agg[:i]), sum(level_agg[:i+1]))]
        for i, level in enumerate(LEVEL_NAMES)
    }

def make_hierarchy_agg_dict(
    df: pd.DataFrame,
    level_counts: List[List[int]]
) -> Dict[int, Tuple[int, List[int]]]:
    """ creates dict of which columns (series) aggregate to which other individual columns (series)
        in the hierarchical, grouped time series 
    
    Arguments:
        df {pd.DataFrame} -- data frame with all series
        level_counts {List[List[int]]} -- list of the counts of leaves that sum to each node, 
            one list for each level in the hierarchy
    
    Returns:
        Dict[int, Tuple[int, List[int]]] -- dictionary with this structure
            constraint number: (aggregate series index, [disaggregated series indices])
    """

    agg = [counts for count_list in level_counts for counts in count_list]
    geo_constraints = {
        s_i: (s_i, [
            idx for idx in range(sum(agg[:s_i+1]), sum(agg[:s_i+1]) + disagg_count)
        ]) 
        for s_i, disagg_count in enumerate(agg[1:])
    }

    # travel_geo_constraints = {
    #     s_i + len(geo_constraints): (s_i, [
    #         i for i in range(
    #             sum(agg) + s_i * TRAVEL_TYPES, 
    #             sum(agg) + (s_i + 1) * TRAVEL_TYPES
    #         )
    #     ])
    #     for s_i in range(sum(agg))
    # }

    travel_constraints = {
        s_i * TRAVEL_TYPES + s_j + len(geo_constraints): (
            s_i * TRAVEL_TYPES + s_j + sum(agg), 
            [idx for idx in range(
                s_j + sum(agg) + sum(agg[:s_i+1]) * TRAVEL_TYPES, 
                s_j + sum(agg) + (sum(agg[:s_i+1]) + disagg_count) * TRAVEL_TYPES,
                TRAVEL_TYPES
            )]
        )
        for s_j in range(TRAVEL_TYPES)
        for s_i, disagg_count in enumerate(agg[1:])
    }

    return {**geo_constraints, **travel_constraints}


def split(
    X: np.ndarray,
    horizon: int = 12,
    min_train_size: int = None,
    max_train_size: int = None,
) -> List[Tuple[List[int]]]:
    """ creates a list of train/test indices for time series cross-fold validation
    
    Arguments:
        X {np.ndarray} -- values to split by index
    
    Keyword Arguments:
        horizon {int} -- length of prediction horizon (default: {12})
        max_train_size {int} -- maximum size for a single training set
            (default: {None})
        min_train_size {int} -- minimum size for a single training set. If None, 
            it will be set to the length of the prediction horizon (default: {None})
    
    Returns:
        List[Tuple[List[int]]] -- list of train/test index tuples
    """

    n_splits = X.shape[0] // horizon - 1
    splits = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)

    if min_train_size is None:
        min_train_size = horizon

    filtered_splits = []
    for train_index, test_index in splits.split(X):
        if len(train_index) >= min_train_size:
            filtered_splits.append((train_index, test_index))
    
    print(f'The dataset has been split into {len(filtered_splits)} folds for CV')
    return filtered_splits

def build_datasets(
    data_df: pd.DataFrame, 
    splits: List[Tuple[List[int]]],
    val: bool = True,
) ->  List[Tuple[ListDataset]]:
    """ creates a list of gluonts train/val(optional)/test tuples, one for each CV fold
    
    Arguments:
        data_df {pd.DataFrame} -- df with data for all series (all levels of hierarchy)
        splits {List[Tuple[List[int]]]} -- list of train/test index tuples
        
    Keyword Arguments:
        val {bool} -- whether to segment part of the training set for validation. If True, 
            last `horizon` length idxs will be used as validation set (default: {True})
    
    Returns:
        List[Tuple[ListDataset]] -- list of gluonts train/test tuples, one for each CV fold
    """

    if val:
        return [
            (ListDataset(
                [
                    {
                        FieldName.START: data_df.iloc[train_idxs[:-len(test_idxs)],:].index[0],
                        FieldName.TARGET: data_df.iloc[train_idxs[:-len(test_idxs)],col_idx].values,
                        FieldName.FEAT_STATIC_CAT: np.array([col_idx]),
                    }
                    for col_idx in range(data_df.shape[1])
                ],
                freq='M'
            ),
            ListDataset(
                [
                    {
                        FieldName.START: data_df.iloc[train_idxs[-len(test_idxs):],:].index[0],
                        FieldName.TARGET: data_df.iloc[train_idxs[-len(test_idxs):],col_idx].values,
                        FieldName.FEAT_STATIC_CAT: np.array([col_idx]),
                    }
                    for col_idx in range(data_df.shape[1])
                ],
                freq='M'
            ),
            ListDataset(
                [
                    {
                        FieldName.START: data_df.iloc[test_idxs,:].index[0],
                        FieldName.TARGET: data_df.iloc[test_idxs,col_idx].values,
                        FieldName.FEAT_STATIC_CAT: np.array([col_idx]),
                    }
                    for col_idx in range(data_df.shape[1])
                ],
                freq='M'
            ))
            for (train_idxs, test_idxs) in splits 
        ]
    else:
        return [
            (ListDataset(
                [
                    {
                        FieldName.START: data_df.iloc[train_idxs,:].index[0],
                        FieldName.TARGET: data_df.iloc[train_idxs,col_idx].values,
                        FieldName.FEAT_STATIC_CAT: np.array([col_idx]),
                    }
                    for col_idx in range(data_df.shape[1])
                ],
                freq='M'
            ),
            ListDataset(
                [
                    {
                        FieldName.START: data_df.iloc[test_idxs,:].index[0],
                        FieldName.TARGET: data_df.iloc[test_idxs,col_idx].values,
                        FieldName.FEAT_STATIC_CAT: np.array([col_idx]),
                    }
                    for col_idx in range(data_df.shape[1])
                ],
                freq='M'
            ))
            for (train_idxs, test_idxs) in splits 
        ]
