# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf import DataFrame
from cudf.core.column import as_column

from cuspatial._lib import spatial_window


def points_in_spatial_window(min_x, max_x, min_y, max_y, xs, ys):
    """ Return only the subset of coordinates that fall within the numerically
    closed borders [,] of the defined bounding box.

    A point (x, y) is inside the query window if and only if
    min_x < x < max_x AND min_y < y < max_y

    params
    min_x: lower x-coordinate of the query window
    max_x: upper x-coordinate of the query window
    min_y: lower y-coordinate of the query window
    max_y: upper y-coordinate of the query window
    
    xs: Series of x-coordinates that may fall within the window
    ys: Series of y-coordinates that may fall within the window

    Parameters
    ----------
    {params}

    Returns
    -------
    DataFrame: subset of x, y pairs above that fall within the window
    """
    result = spatial_window.points_in_spatial_window(min_x, max_x,
                                                     min_y, max_y,
                                                     as_column(xs),
                                                     as_column(ys))
    return DataFrame._from_table(result)
