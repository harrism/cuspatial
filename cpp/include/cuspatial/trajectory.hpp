/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/types.h>

namespace cuspatial {

/**
 * @brief derive trajectories from points, timestamps and object ids
 *
 * Points are x/y coordinates relative to an origin. First sorts by object id
 * and timestamp and then groups by id.
 *
 * @param[in/out] x: x coordinates relative to a camera origin
 *                  (before/after sorting)
 * @param[in/out] y: y coordinates relative to a camera origin
 *                   (before/after sorting)
 * @param[in/out] object_id: object (e.g., vehicle) id column (before/after
 *                sorting); upon completion, unique ids become trajectory ids
 * @param[in/out] timestamp: timestamp column (before/after sorting)
 * @param[out] trajectory_id: trajectory id column (see comments on oid)
 * @param[out] length: #of points in the derived trajectories
 * @param[out] offset: position offsets of trajectories used to index x, y,
 *                  object_id and timestamp
 * 
 * @return number of derived trajectories
 */
int derive_trajectories(gdf_column& x, gdf_column& y, gdf_column& object_id,
                        gdf_column& timestamp, gdf_column& trajectory_id,
                        gdf_column& length, gdf_column& offset);


/**
 * @brief Compute the distance and speed of trajectories
 *
 * Trajectories are typically derived from coordinate data using
 * derive_trajectories().
 * 
 * @param[in] x: x coordinates relative to a camera origin and ordered by
 *            (id,timestamp)
 * @param[in] y: y coordinates relative to a camera origin and ordered by
 *            (id,timestamp)
 * @param[in] timestamp: timestamp column ordered by (id,timestamp)
 * @param[in] length: number of points column ordered by (id,timestamp)
 * @param[in] offset: offsets of trajectories used to index x/y/oid/ts
 *            ordered by (id,timestamp)
 * @param[out] dist: computed distances/lengths of trajectories in meters (m)
 * @param[out] speed: computed speed of trajectories in meters per second (m/s)
 *
 * Note: May output duration in the future (in addition to distance/speed)
 * if needed. Duration can be computed on CPU by fetching begining/ending
 * timestamps of a trajectory in the timestamp array
 */
std::pair<gdf_column,gdf_column>
trajectory_distance_and_speed(const gdf_column& x, const gdf_column& y,
                              const gdf_column& timestamp,
                              const gdf_column& length,
                              const gdf_column& offset);


/**
 * @brief compute spatial bounding boxes of trajectories
 *
 * @param[in] x: x coordinates relative to a camera origin and ordered by
 *            (id, timestamp)
 * @param[in] y: y coordinates relative to a camera origin and ordered by
 *            (id, timestamp)
 * @param[in] length: number of points column ordered by (id, timestamp)
 * @param[in] offset: offsets of trajectories used to index x/y ordered by
 *            (id,timestamp)
 * @param[out] bbox_x1: x coordinates of the lower-left corners of computed
 *             spatial bounding boxes
 * @param[out] bbox_y1: y coordinates of the lower-left corners of computed
 *             spatial bounding boxes
 * @param[out] bbox_x2: x coordinates of the upper-right corners of computed
 *             spatial bounding boxes
 * @param[out] bbox_y2: y coordinates of the upper-right corners of computed
 *             spatial bounding boxes
 *
 * Note: temporal 1D bounding box can be computed similary but it seems that
 * there is no such a need; Similar to the discussion in derive_trajectories,
 * temporal 1D bounding box can be retrieved directly
 */
void trajectory_spatial_bounds(const gdf_column& x, const gdf_column& y,
                               const gdf_column& length,
                               const gdf_column& offset,
                               gdf_column& bbox_x1, gdf_column& bbox_y1,
                               gdf_column& bbox_x2, gdf_column& bbox_y2);

/**
 * @brief Return a subset of trajectories selected by ID

 * @param[in] id: ids of trajectories whose x/y/len/pos data will be kept
 * @param[in] in_x: input x coordinates
 * @param[in] in_y: input y coordinates
 * @param[in] in_id: input ids of points
 * @param[in] in_timestamp: input timestamps of points
 * @param[out] out_x: output x coordinates ordered by (in_id,in_ts)
 * @param[out] out_y: output y coordinates ordered by (in_id,in_ts)
 * @param[out] out_id: output ids ordered by (in_id,in_ts)
 * @param[out] out_timestamp: output timestamp ordered by (in_id,in_ts)
 * 
 * @return number of trajectories returned
 * 
 * @note the output columns are allocated by this function but they must 
 * be deallocated by the caller.
 * 
 * @note this function is likely to be removed in the future since it is 
 * redundant to cuDF functionality
 *
 * @note: the API is useful for integrating with cuDF and serial Python APIs,
 * e.g., query based on trajectory level information using serial Python APIs or
 * cuDF APIs and identify a subset of trajectory IDs. These IDs can then be used
 * to retrieve x/y/len/pos data for futher processing.
 */
gdf_size_type subset_trajectory_id(const gdf_column& id,
                                   const gdf_column& in_x,
                                   const gdf_column& in_y,
                                   const gdf_column& in_id,
                                   const gdf_column& in_timestamp,
                                   gdf_column& out_x,
                                   gdf_column& out_y,
                                   gdf_column& out_id,
                                   gdf_column& out_timestamp);

}  // namespace cuspatial