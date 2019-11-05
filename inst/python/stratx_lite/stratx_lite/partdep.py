import numpy as np
import pandas as pd
from typing import Mapping, List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings


def leaf_samples(rf, X:np.ndarray):
    """
    Return a list of arrays where each array is the set of X sample indexes
    residing in a single leaf of some tree in rf forest.
    """
    ntrees = len(rf.estimators_)
    leaf_ids = rf.apply(X) # which leaf does each X_i go to for each tree?
    d = pd.DataFrame(leaf_ids, columns=[f"tree{i}" for i in range(ntrees)])
    d = d.reset_index() # get 0..n-1 as column called index so we can do groupby
    """
    d looks like:
        index	tree0	tree1	tree2	tree3	tree4
    0	0	    8	    3	    4	    4	    3
    1	1	    8	    3	    4	    4	    3
    """
    leaf_samples = []
    for i in range(ntrees):
        """
        Each groupby gets a list of all X indexes associated with same leaf. 4 leaves would
        get 4 arrays of X indexes; e.g.,
        array([array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
               array([10, 11, 12, 13, 14, 15]), array([16, 17, 18, 19, 20]),
               array([21, 22, 23, 24, 25, 26, 27, 28, 29]), ... )
        """
        sample_idxs_in_leaf = d.groupby(f'tree{i}')['index'].apply(lambda x: x.values)
        leaf_samples.extend(sample_idxs_in_leaf) # add [...sample idxs...] for each leaf
    return leaf_samples


def collect_point_betas(X, y, colname, leaves, nbins:int):
    ignored = 0
    leaf_xranges = []
    leaf_slopes = []
    point_betas = np.full(shape=(len(X),), fill_value=np.nan)

    for samples in leaves: # samples is set of obs indexes that live in a single leaf
        leaf_all_x = X.iloc[samples]
        leaf_x = leaf_all_x[colname].values
        leaf_y = y.iloc[samples].values
        # Right edge of last bin is max(leaf_x) but that means we ignore the last value
        # every time. Tweak domain right edge a bit so max(leaf_x) falls in last bin.
        last_bin_extension = 0.0000001
        domain = (np.min(leaf_x), np.max(leaf_x)+last_bin_extension)
        bins = np.linspace(*domain, num=nbins+1, endpoint=True)
        binned_idx = np.digitize(leaf_x, bins) # bin number for values in leaf_x
        for b in range(1, len(bins)+1):
            bin_x = leaf_x[binned_idx == b]
            bin_y = leaf_y[binned_idx == b]
            if len(bin_x) < 2: # could be none or 1 in bin
                ignored += len(bin_x)
                continue
            r = (np.min(bin_x), np.max(bin_x))
            if len(bin_x)<2 or np.isclose(r[0], r[1]):
                ignored += len(bin_x)
                continue
            lm = LinearRegression()
            leaf_obs_idx_for_bin = np.nonzero((leaf_x>=bins[b-1]) &(leaf_x<bins[b]))
            obs_idx = samples[leaf_obs_idx_for_bin]
            lm.fit(bin_x.reshape(-1, 1), bin_y)
            point_betas[obs_idx] = lm.coef_[0]
            leaf_slopes.append(lm.coef_[0])
            leaf_xranges.append(r)

    leaf_slopes = np.array(leaf_slopes)
    return leaf_xranges, leaf_slopes, point_betas, ignored


def stratpd(X, y, colname, targetname,
            ntrees=1, min_samples_leaf=10, bootstrap=False, max_features=1.0
            ):
    
    # For compatibility with reticulate
    if y.ndim == 2:
        y = y.iloc[:, 0]

    rf = RandomForestRegressor(n_estimators=ntrees,
                               min_samples_leaf=min_samples_leaf,
                               bootstrap=bootstrap,
                               max_features=max_features)
    rf.fit(X.drop(colname, axis=1), y)

    leaves = leaf_samples(rf, X.drop(colname, axis=1))
    nnodes = rf.estimators_[0].tree_.node_count

    leaf_xranges, leaf_sizes, leaf_slopes, ignored = \
        collect_discrete_slopes(rf, X, y, colname)

    real_uniq_x = np.array(sorted(np.unique(X[colname])))

    slope_at_x = avg_values_at_x(real_uniq_x, leaf_xranges, leaf_slopes)

    # Drop any nan slopes; implies we have no reliable data for that range
    # Make sure to drop uniq_x values too :)
    notnan_idx = ~np.isnan(slope_at_x) # should be same for slope_at_x and r2_at_x
    slope_at_x = slope_at_x[notnan_idx]
    pdpx = real_uniq_x[notnan_idx]

    dx = np.diff(pdpx)
    y_deltas = slope_at_x[:-1] * dx  # last slope is nan since no data after last x value
    pdpy = np.cumsum(y_deltas)                    # we lose one value here
    pdpy = np.concatenate([np.array([0]), pdpy])  # add back the 0 we lost

    return leaf_xranges, leaf_slopes, pdpx, pdpy, ignored


def discrete_xc_space(x: np.ndarray, y: np.ndarray, colname):
    """
    Use the categories within a leaf as the bins to dynamically change the bins,
    rather then using a fixed nbins hyper parameter. Group the leaf x,y by x
    and collect the average y.  The unique x and y averages are the new x and y pairs.
    The slope for each x is:

        (y_{i+1} - y_i) / (x_{i+1} - x_i)

    If the ordinal/ints are exactly one unit part, then it's just y_{i+1} - y_i. If
    they are not consecutive, we do not ignore isolated x_i as it ignores too much data.
    E.g., if x is [1,3,4] and y is [9,8,10] then the second x coordinate is skipped.
    The two slopes are [(8-9)/2, (10-8)/1] and bin widths are [2,1].

    If there is exactly one category in the leaf, the leaf provides no information
    about how the categories contribute to changes in y. We have to ignore this leaf.
    """
    ignored = 0
    xy = pd.concat([pd.Series(x), pd.Series(y)], axis=1)
    xy.columns = ['x', 'y']
    xy = xy.sort_values('x')
    df_avg = xy.groupby('x').mean().reset_index()
    x = df_avg['x'].values
    y = df_avg['y'].values
    uniq_x = x

    if len(uniq_x)==1:
        # print(f"ignore {len(x)} in discrete_xc_space")
        ignored += len(x)
        return np.array([]), np.array([]), np.array([]), np.array([]), ignored

    bin_deltas = np.diff(uniq_x)
    y_deltas = np.diff(y)
    leaf_slopes = y_deltas / bin_deltas  # "rise over run"
    leaf_xranges = np.array(list(zip(uniq_x, uniq_x[1:])))
    leaf_sizes = xy['x'].value_counts().sort_index().values

    return leaf_xranges, leaf_sizes, leaf_slopes, [], ignored


def collect_discrete_slopes(rf, X, y, colname):
    """
    For each leaf of each tree of the random forest rf (trained on all features
    except colname), get the samples then isolate the column of interest X values
    and the target y values. Perform another partition of X[colname] vs y and do
    piecewise linear regression to get the slopes in various regions of X[colname].
    We don't need to subtract the minimum y value before regressing because
    the slope won't be different. (We are ignoring the intercept of the regression line).

    Return for each leaf, the ranges of X[colname] partitions, num obs per leaf,
    associated slope for each range

    Only does discrete now after doing pointwise continuous slopes differently.
    """
    leaf_slopes = []  # drop or rise between discrete x values
    leaf_xranges = [] # drop is from one discrete value to next
    leaf_sizes = []

    ignored = 0

    leaves = leaf_samples(rf, X.drop(colname, axis=1))

    for samples in leaves:
        one_leaf_samples = X.iloc[samples]
        leaf_x = one_leaf_samples[colname].values
        leaf_y = y.iloc[samples].values

        r = (np.min(leaf_x), np.max(leaf_x))
        if np.isclose(r[0], r[1]):
            # print(f"ignoring xleft=xright @ {r[0]}")
            ignored += len(leaf_x)
            continue

        leaf_xranges_, leaf_sizes_, leaf_slopes_, leaf_r2_, ignored_ = \
            discrete_xc_space(leaf_x, leaf_y, colname=colname)

        leaf_slopes.extend(leaf_slopes_)
        leaf_xranges.extend(leaf_xranges_)
        leaf_sizes.extend(leaf_sizes_)
        ignored += ignored_

    leaf_xranges = np.array(leaf_xranges)
    leaf_sizes = np.array(leaf_sizes)
    leaf_slopes = np.array(leaf_slopes)

    return leaf_xranges, leaf_sizes, leaf_slopes, ignored


def avg_values_at_x(uniq_x, leaf_ranges, leaf_values):
    """
    Compute the weighted average of leaf_values at each uniq_x.

    Value at max(x) is NaN since we have no data beyond that point.
    """
    nx = len(uniq_x)
    nslopes = len(leaf_values)
    slopes = np.zeros(shape=(nx, nslopes))
    i = 0  # leaf index; we get a line for each leaf
    # collect the slope for each range (taken from a leaf) as collection of
    # flat lines across the same x range
    for r, slope in zip(leaf_ranges, leaf_values):
        s = np.full(nx, slope, dtype=float)
        # now trim line so it's only valid in range r;
        # don't set slope on right edge
        s[np.where( (uniq_x < r[0]) | (uniq_x >= r[1]) )] = np.nan
        slopes[:, i] = s
        i += 1
    # The value could be genuinely zero so we use nan not 0 for out-of-range
    # Now average horiz across the matrix, averaging within each range
    # Wrap nanmean() in catcher to avoid "Mean of empty slice" warning, which
    # comes from some rows being purely NaN; I should probably look at this sometime
    # to decide whether that's hiding a bug (can there ever be a nan for an x range)?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_value_at_x = np.nanmean(slopes, axis=1)

    return avg_value_at_x
