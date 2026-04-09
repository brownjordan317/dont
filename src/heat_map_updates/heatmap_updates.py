import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


DECAY_REFERENCE_SECONDS = 0.1


class HeatMapUpdates:
    def __init__(self, heatmap, search_decay_percent_per_100ms=100.0):
        if isinstance(heatmap, np.ndarray):
            self.data = heatmap.astype(np.float32)
            self.origin = (0.0, 0.0)
            self.resolution = 1.0
        else:
            self.data = heatmap["data"].astype(np.float32)
            self.origin = heatmap["origin"]
            self.resolution = heatmap["resolution"]

        self.H, self.W = self.data.shape
        self.total_heat = float(self.data.sum(dtype=np.float64))
        self._cluster_state = None
        self.search_decay_percent_per_100ms = float(search_decay_percent_per_100ms)

    # -----------------------------------------------------

    def change_to_zeroes(self, projection, dt_seconds=DECAY_REFERENCE_SECONDS):
        poly = self._projection_to_pixel_poly(projection)

        col_min = max(int(np.floor(poly[:, 0].min())), 0)
        col_max = min(int(np.ceil(poly[:, 0].max())), self.W - 1)
        row_min = max(int(np.floor(poly[:, 1].min())), 0)
        row_max = min(int(np.ceil(poly[:, 1].max())), self.H - 1)

        if row_min > row_max or col_min > col_max:
            return self

        mask = self._roi_polygon_mask(poly, row_min, row_max, col_min, col_max)
        if not np.any(mask):
            return self

        roi = self.data[row_min:row_max + 1, col_min:col_max + 1]
        decay_amount = self._resolve_decay_amount(dt_seconds)
        if decay_amount <= 0.0:
            return self

        # Search persistence is modeled as a subtractive decay on the normalized
        # heatmap. A setting of 100 removes the whole covered value in one
        # 100 ms observation slice, while smaller settings let repeatedly
        # observed cells fade out over time instead of snapping to zero.
        current_values = roi[mask]
        if decay_amount >= 1.0:
            removed_heat = float(current_values.sum(dtype=np.float64))
            roi[mask] = 0.0
        else:
            updated_values = np.maximum(current_values - decay_amount, 0.0)
            removed_heat = float((current_values - updated_values).sum(dtype=np.float64))
            roi[mask] = updated_values

        self.total_heat = max(0.0, self.total_heat - removed_heat)

        self._mark_dirty_blocks(row_min, row_max + 1, col_min, col_max + 1)
        return self

    # -----------------------------------------------------

    def cluster_heatmap(self, cluster_size=100):
        state = self._ensure_cluster_state(cluster_size)
        if state is None:
            return self

        self._sync_dirty_blocks(state)
        return self

    def get_cluster_view(self, cluster_size=100):
        state = self._ensure_cluster_state(cluster_size)
        if state is None:
            return None

        self._sync_dirty_blocks(state)
        return state["display_means"]

    # -----------------------------------------------------

    def get_top_clusters(self, cluster_size=100, top_k=5):
        state = self._ensure_cluster_state(cluster_size)
        if state is None:
            return []

        self._sync_dirty_blocks(state)

        means = state["display_means"]
        flat = means.reshape(-1)

        positive = np.flatnonzero(flat > 0)
        if positive.size == 0:
            return []

        count = min(top_k, positive.size)

        top_flat_indices = positive[
            np.argpartition(flat[positive], -count)[-count:]
        ]

        ordered_indices = top_flat_indices[
            np.argsort(flat[top_flat_indices])[::-1]
        ]

        results = []
        for flat_index in ordered_indices:
            r, c = divmod(flat_index, means.shape[1])
            bounds = self._cluster_bounds(state, r, c)
            results.append((*bounds, float(flat[flat_index])))

        return results

    def find_hottest_cluster(self, cluster_size=100):
        top_clusters = self.get_top_clusters(cluster_size, top_k=1)
        if not top_clusters:
            return None
        return top_clusters[0][:4]

    # -----------------------------------------------------

    def _utm_to_pixel(self, easting, northing):
        col = (easting - self.origin[0]) / self.resolution
        row = (self.origin[1] - northing) / self.resolution
        return int(col), int(row)

    def _projection_to_pixel_poly(self, projection):
        corners = (
            self._utm_to_pixel(*projection["tl"]),
            self._utm_to_pixel(*projection["tr"]),
            self._utm_to_pixel(*projection["br"]),
            self._utm_to_pixel(*projection["bl"]),
        )
        return np.array(corners, dtype=np.int32)

    # -----------------------------------------------------

    @staticmethod
    def _roi_polygon_mask(poly, row_min, row_max, col_min, col_max):
        if cv2 is not None:
            shifted = poly.copy()
            shifted[:, 0] -= col_min
            shifted[:, 1] -= row_min

            mask = np.zeros(
                (row_max - row_min + 1, col_max - col_min + 1),
                dtype=np.uint8
            )
            cv2.fillConvexPoly(mask, shifted.astype(np.int32), 1)
            return mask.view(bool)

        xs = np.arange(col_min, col_max + 1, dtype=np.float32) + 0.5
        ys = np.arange(row_min, row_max + 1, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)

        vertices = poly.astype(np.float32)
        next_vertices = np.roll(vertices, -1, axis=0)

        edge_x = next_vertices[:, 0] - vertices[:, 0]
        edge_y = next_vertices[:, 1] - vertices[:, 1]

        rel_x = grid_x[None] - vertices[:, 0][:, None, None]
        rel_y = grid_y[None] - vertices[:, 1][:, None, None]

        cross = edge_x[:, None, None] * rel_y - edge_y[:, None, None] * rel_x
        eps = 1e-6

        return np.all(cross >= -eps, axis=0) | np.all(cross <= eps, axis=0)

    def _resolve_decay_amount(self, dt_seconds):
        dt_seconds = max(0.0, float(dt_seconds))
        decay_percent = float(np.clip(self.search_decay_percent_per_100ms, 0.0, 100.0))
        return (decay_percent / 100.0) * (dt_seconds / DECAY_REFERENCE_SECONDS)

    # -----------------------------------------------------

    def _ensure_cluster_state(self, cluster_size):
        block = max(1, int(cluster_size / self.resolution))

        state = self._cluster_state
        if state is not None and state["block"] == block:
            return state

        h_trim = (self.H // block) * block
        w_trim = (self.W // block) * block

        if h_trim == 0 or w_trim == 0:
            self._cluster_state = None
            return None

        trimmed = self.data[:h_trim, :w_trim]

        means = self._compute_cluster_means(trimmed, block)

        self._cluster_state = {
            "block": block,
            "H_trim": h_trim,
            "W_trim": w_trim,
            "means": means,
            "display_means": self._to_display_means(means),
            "dirty": np.zeros((h_trim // block, w_trim // block), dtype=bool),
        }

        return self._cluster_state

    # -----------------------------------------------------

    def _sync_dirty_blocks(self, state):
        dirty_rows, dirty_cols = np.nonzero(state["dirty"])
        if dirty_rows.size == 0:
            return

        mean_values = self._compute_dirty_block_means(
            state, dirty_rows, dirty_cols
        )

        means = state["means"]
        display_means = state["display_means"]

        for idx, (r, c) in enumerate(zip(dirty_rows, dirty_cols)):
            val = float(mean_values[idx])
            means[r, c] = val
            display_means[r, c] = val

        state["dirty"][dirty_rows, dirty_cols] = False

    # -----------------------------------------------------

    def _mark_dirty_blocks(self, row_start, row_stop, col_start, col_stop):
        state = self._cluster_state
        if state is None:
            return

        row_start = max(row_start, 0)
        col_start = max(col_start, 0)
        row_stop = min(row_stop, state["H_trim"])
        col_stop = min(col_stop, state["W_trim"])

        if row_start >= row_stop or col_start >= col_stop:
            return

        block = state["block"]

        r0 = row_start // block
        r1 = (row_stop - 1) // block + 1
        c0 = col_start // block
        c1 = (col_stop - 1) // block + 1

        state["dirty"][r0:r1, c0:c1] = True

    # -----------------------------------------------------

    @staticmethod
    def _cluster_bounds(state, block_row, block_col):
        row_min = block_row * state["block"]
        col_min = block_col * state["block"]
        row_max = row_min + state["block"]
        col_max = col_min + state["block"]
        return row_min, col_min, row_max, col_max

    # -----------------------------------------------------

    def _compute_cluster_means(self, trimmed, block):
        h_trim, w_trim = trimmed.shape

        return trimmed.reshape(
            h_trim // block,
            block,
            w_trim // block,
            block
        ).mean(axis=(1, 3)).astype(np.float32, copy=False)

    # -----------------------------------------------------

    def _compute_dirty_block_means(self, state, dirty_rows, dirty_cols):
        block = state["block"]

        mean_values = np.empty(dirty_rows.size, dtype=np.float32)

        for idx, (r, c) in enumerate(zip(dirty_rows, dirty_cols)):
            row_min, col_min, row_max, col_max = self._cluster_bounds(
                state, r, c
            )
            mean_values[idx] = self.data[
                row_min:row_max,
                col_min:col_max
            ].mean()

        return mean_values

    # -----------------------------------------------------

    def _to_display_means(self, means):
        return means
