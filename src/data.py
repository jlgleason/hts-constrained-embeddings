# Standard library imports
from typing import Tuple, List, Dict, Iterable
import re
import pandas as pd
import numpy as np

# Third Party imports
from sklearn.model_selection import TimeSeriesSplit
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from gluonts.transform import InstanceSampler
from gluonts.core.component import validated

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
        re.compile(f'^{region}.+') 
        for region in prefix_3
    ]
    zone_labels = [
        re.compile(f'^{zone}.+') 
        for zone in prefix_2
    ]
    state_labels = [
        re.compile(f'^{state}.+') 
        for state in prefix_1
    ] 
    zone_by_travel_labels = [
        re.compile(f'^{zone}.{travel}')
        for zone in prefix_2
        for travel in travel_types
    ]
    state_by_travel_labels = [
        re.compile(f'^{state}..{travel}')
        for state in prefix_1
        for travel in travel_types
    ]
    country_by_travel_labels = [
        re.compile(f'^...{travel}') 
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
) -> Tuple[pd.DataFrame, List[List[int]]]:
    """ sums df columns that match regular expressions in 'filter_cols' along column 
        (series, not time) axis, sorts columns of resulting df lexicographically
    
    Arguments:
        df {pd.DataFrame} -- data frame
        filter_col_names {List[re.Pattern]} -- roots of strings that should match for summation
        
    Keyword Arguments:
        sort {bool} -- whether to sort columns of df lexicographically before returning 
            (default: {True})
    
    Returns:
        Tuple[pd.DataFrame, List[List[int]] -- 
            1) summed and (optionally) lexicographically sorted data frame
            2) nested list indices of df columns that match each regular expression
    """

    new_df = pd.DataFrame(index = df.index)
    all_match_idxs = []
    for regex in filter_cols:
        match_idxs = [i for i, col in enumerate(df.columns) if re.search(regex, col)]
        all_match_idxs.append(match_idxs)
        new_df[regex.pattern] = df.iloc[:, match_idxs].sum(axis=1)
    if sort:
        new_df = new_df.reindex(sorted(new_df.columns), axis = 1)
    return new_df, all_match_idxs

def all_dfs(
    tourism_df: pd.DataFrame,
    labels: List[List[re.Pattern]]
) -> Tuple[pd.DataFrame, List[int]]:
    """ generate and combine dataframes for all levels of the hierarchy

    Arguments:
        tourism_df {pd.DataFrame} -- tourism frame with bottom level series (region/type)
        labels {List[List[re.Pattern]]} -- location regexes (one list for each level of hierarchy)

    Returns:
        Tuple[pd.DataFrame, List[int] -- 
            1) single combined dataframe
            2) oredered list of indices of base forecast df columns that match 
                aggregated forecast column
    """

    region_labels, zone_labels, state_labels, zone_by_travel_labels, state_by_travel_labels, country_by_travel_labels = labels

    region_df, region_idxs = sum_filtered_cols(tourism_df, region_labels)
    zone_df, zone_idxs = sum_filtered_cols(tourism_df, zone_labels)
    state_df, state_idxs = sum_filtered_cols(tourism_df, state_labels)
    zone_by_travel_df, zone_by_travel_idxs = sum_filtered_cols(
        tourism_df, 
        zone_by_travel_labels,
        sort=False    
    )
    state_by_travel_df, state_by_travel_idxs = sum_filtered_cols(
        tourism_df, 
        state_by_travel_labels, 
        sort=False
    )
    country_df, country_idxs = sum_filtered_cols(tourism_df, [re.compile(f'.+')])
    country_by_travel_df, country_by_travel_idxs = sum_filtered_cols(
        tourism_df, 
        country_by_travel_labels,
        sort=False
    )
    all_df = pd.concat(
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

    all_idxs = country_idxs + state_idxs + zone_idxs + region_idxs + country_by_travel_idxs + \
        state_by_travel_idxs + zone_by_travel_idxs

    all_idxs = [
        [idx + all_df.shape[1] - tourism_df.shape[1] for idx in idx_list]
        for idx_list in all_idxs
    ]
    return all_df, all_idxs

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
) -> Tuple[pd.DataFrame, List[int], List[List[int]]]:
    """ preprocess hierarchical tourism df into one df with all level series
    
    Arguments:
        tourism_df {pd.DataFrame} -- tourism frame with bottom level series (region/type)
    
    Returns:
        Tuple[pd.DataFrame, List[int], List[List[int]]] -- 
            1) new df with region, zone, state, and country series as separate columns
            2) list of the counts of leaves that sum to each node in the hierarchy
            3) list of list of indices representing the base level series that add up 
                to each aggregation constraint
    """

    tourism_df['Year'] = tourism_df['Year'].ffill().astype(int)
    tourism_df['Date'] = tourism_df['Month'] + ' ' + tourism_df['Year'].astype(str)
    tourism_df = tourism_df.drop(columns = ['Year', 'Month'])
    tourism_df = tourism_df.set_index('Date')

    travel_types = [col[-3:] for col in tourism_df.columns[:4]]
    prefixes = unique_prefixes(tourism_df.columns, [1,2,3])
    labels = regex_labels(travel_types, prefixes)
    all_df, prefix_idxs = all_dfs(tourism_df, labels)
    counts = level_counts(prefixes, labels)

    return all_df, counts, prefix_idxs

def make_hierarchy_level_dict(
    level_counts: List[List[int]]
) -> Dict[str, List[int]]:
    """ creates dict of which columns (series) belong to which level of hierarchy 
    
    Arguments:
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
    prefix_idxs: List[List[int]],
) -> Dict[int, Tuple[int, List[int]]]:
    """ creates dict of which columns (series) aggregate to which other individual columns (series)
        in the hierarchical, grouped time series 
    
    Arguments:
        level_counts {List[List[int]]} -- list of the counts of leaves that sum to each node, 
            one list for each level in the hierarchy
        prefix_idxs {List[List[int]]} -- list of list of indices representing the base level series that add up 
            to each aggregation constraint
    
    Returns:
        Dict[int, Tuple[int, List[int]]] -- dictionary with this structure
            constraint number: (aggregate series index, [disaggregated series indices])
    """

    return {agg_idx: base_idxs for agg_idx, base_idxs in enumerate(prefix_idxs)}

def inspect_hierarchy_agg_dict(
    hierarchy_agg_dict: Dict[int, Tuple[int, List[int]]],
    column_names: pd.Index,
    idx: int = 0
) -> None:
    """ prints key/value pair in hierarchy agg dict and names of corresponding columns in df

    Arguments:
        hierarchy_agg_dict {Dict[int, Tuple[int, List[int]]]} -- mapping of which columns (series) 
            aggregate to which other individual columns (series) in the hierarchical, grouped time series
        column_names {pd.Index} -- list of columns
        
    Keyword Arguments:
        idx {int} -- idx of key in hierarchy to inspect (default: {0})
    """
    print(f'Aggregate column: {column_names[idx]}')
    print(f'Disaggregated columns: {column_names[hierarchy_agg_dict[idx]]}')

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
            filtered_splits.append((train_index, test_index[:horizon]))
    
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