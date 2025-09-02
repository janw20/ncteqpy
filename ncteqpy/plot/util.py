from __future__ import annotations

import time

from matplotlib import _api
from matplotlib.legend import Legend
from matplotlib.transforms import Bbox


class AdditionalLegend(Legend):

    order: int

    def __init__(self, order: int, *args, **kwargs) -> None:
        self.order = order
        super().__init__(*args, **kwargs)

    def _find_best_position(self, width, height, renderer):
        """Determine the best location to place the legend."""
        assert self.isaxes  # always holds, as this is only called internally

        start_time = time.perf_counter()

        bboxes, lines, offsets = self._auto_legend_data(renderer)

        bbox = Bbox.from_bounds(0, 0, width, height)

        # get all legends added before this one
        legends = [
            l for l in self.axes.get_children() if isinstance(l, AdditionalLegend)
        ]  # TODO: make this work with figure

        # print([l.texts[0] for l in legends])
        # print([legend is self for legend in legends])

        # sort legends by size to place larger legends first
        legends = sorted(legends, key=lambda l: l.order)

        i_self = legends.index(self)

        # print("test")

        all_legends = legends[:i_self]
        if l := self.axes.get_legend():
            all_legends.insert(0, l)

        # for i, legend in enumerate(all_legends):
        candidates = []
        for idx in range(1, len(self.codes)):
            l, b = self._get_anchored_bbox(
                idx, bbox, self.get_bbox_to_anchor(), renderer
            )
            legendBox = Bbox.from_bounds(l, b, width, height)
            # XXX TODO: If markers are present, it would be good to take them
            # into account when checking vertex overlaps in the next line.

            # infinite badness if this legend overlaps a larger one
            if any(legendBox.overlaps(l.get_window_extent()) for l in all_legends):
                badness = float("inf")
            else:
                badness = (
                    sum(legendBox.count_contains(line.vertices) for line in lines)
                    + legendBox.count_contains(offsets)
                    + legendBox.count_overlaps(bboxes)
                    + sum(
                        line.intersects_bbox(legendBox, filled=False) for line in lines
                    )
                )
            # Include the index to favor lower codes in case of a tie.
            candidates.append((badness, idx, (l, b)))
            if badness == 0:
                break

            _, _, (l, b) = min(candidates)

            # if i != len(legends) - 1:
            #     legend.set_loc = (l, b)

        else:
            l, b = self._get_anchored_bbox(1, bbox, self.get_bbox_to_anchor(), renderer)

        if self._loc_used_default and time.perf_counter() - start_time > 1:
            _api.warn_external(
                'Creating legend with loc="best" can be slow with large '
                "amounts of data."
            )

        return l, b
