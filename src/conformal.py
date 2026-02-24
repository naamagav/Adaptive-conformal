# Adapted from experiments-conformal-superres by Eduardo Adame, Daniel Csillag, Guilherme Tegoni Goedert
# Core conformal calibration implementation with dynamic programming and bisection search
# for computing risk-controlling thresholds in super-resolution

from typing import Self, Literal
from collections.abc import Sequence
from dataclasses import dataclass
import os
from functools import reduce
import time

from tqdm import tqdm  # type: ignore
from icecream import ic  # type: ignore
import numpy as np  # type: ignore
from jaxtyping import Float, Bool  # type: ignore
from imageio import imread  # type: ignore
from skimage.transform import rescale  # type: ignore
import skimage.color  # type: ignore
from diffusers import LDMSuperResolutionPipeline  # type: ignore
from scipy.signal import convolve2d  # type: ignore
from numba import njit, types  # type: ignore
from numba.typed import Dict

from src.base_model import BaseModel


@dataclass
class ConformalCalibratedModel:
    base_model: BaseModel | None
    thresholds: dict[float, float]  # alpha => threshold

    @classmethod
    def calibrate(
        cls,
        base_model: BaseModel | None,
        data: Sequence[
            tuple[
                Float[np.ndarray, "w h c"] , # low res
                Float[np.ndarray, "w2 h2 c"] ,  # high res
            ],
        ] | None,
        preds_and_prob_masks: Sequence[
            tuple[
                Float[np.ndarray, "w2 h2 c"] | None,  # pred
                Float[np.ndarray, "w2 h2 c"],  # prob mask
            ],
        ],
        *,
        alphas: list[float],
        kernel_size: int = 5,
        diffs: list[np.ndarray] | None = None,
        method: Literal["dynprog"] | Literal["bisection"],
    ) -> Self:
        # Risk:
        #   R(true_image, (predicted_image, confident_mask))
        #   = sup_(p in confident_mask) || true_image[p] - predicted_image[p] ||_1
        # Note that this risk is always at most 3, assuming the colors are RGB in [0, 1] space.

        # We want to find the score
        #   sup { t in RR : 1/(n+1) sum_(i=1)^n R(Y_i, f(X_i; t)) + M/(n+1) <= alpha }
        #   = sup { t in RR : 1/(n+1) sum_(i=1)^n sup_(p in mask_i) ||Y_i[p] - hat(Y)_i[p]||_1 + 3/(n+1) <= alpha }


        preds, prob_masks = zip(*preds_and_prob_masks)

        n = len(prob_masks)

        if diffs is None:
            diffs = [
                convolve2d(
                    np.sum(np.abs(true_image - pred_image), axis=-1),
                    np.ones((kernel_size, kernel_size)) / kernel_size**2,
                    mode="same",
                )
                for (_, true_image), pred_image in tqdm(zip(data, preds), desc="diffs")
            ]

        if method == "dynprog":
            thresholds_to_consider = np.unique(
                np.concatenate([np.ravel(prob_mask) for prob_mask in prob_masks])
            )[::-1]

            data_structure = []
            for diff, prob_mask in tqdm(zip(diffs, prob_masks)):
                assert diff.shape == prob_mask.shape
                prob_mask = np.ravel(prob_mask)
                diff = np.ravel(diff)
                order = np.argsort(prob_mask)[::-1]
                data_structure.append((0, prob_mask[order], diff[order], 0.0))

            @njit
            def dynamic_programming_search(
                data_structure: list[tuple[int, np.ndarray, np.ndarray, float]],
                alphas: list[float],
            ):
                selected_thresholds: dict[float, float] = {
                    alpha: np.inf for alpha in alphas
                }
                progress_n = len(thresholds_to_consider)
                progress_k = max(progress_n // 10_000_000, 1)
                for progress_i, threshold in enumerate(thresholds_to_consider):
                    for i, (init_k, prob_mask, diff, risk) in enumerate(data_structure):
                        m = len(prob_mask)
                        k = init_k
                        subrisk = 0
                        while k < m and prob_mask[k] > threshold:
                            subrisk = max(subrisk, diff[k])
                            k += 1

                        data_structure[i] = (
                            k,
                            prob_mask,
                            diff,
                            max(risk, subrisk),
                        )

                    risk_sum = sum(
                        [risk for _k, _prob_mask, _diff, risk in data_structure]
                    )
                    conformal_risk = risk_sum / (n + 1) + 3 / (n + 1)
                    for alpha in selected_thresholds.keys():
                        if conformal_risk <= alpha:
                            selected_thresholds[alpha] = threshold

                    if progress_i % 1000 * progress_k == 0:
                        print(
                            f"Iteration {progress_i}/{progress_n} ({round(100 * progress_i / progress_n)}%)"
                        )
                print("Done!")

                return selected_thresholds

            print("before")
            before = time.time()
            selected_thresholds = dynamic_programming_search(data_structure, alphas)
            after = time.time()
            print(f"after: {after - before}")

            return cls(
                base_model=base_model,
                thresholds={float(k): float(v) for k, v in selected_thresholds.items()},
            )
        elif method == "bisection":
            all_thresholds = np.concatenate(
                [np.ravel(prob_mask) for prob_mask in prob_masks]
            )
            min_threshold = np.min(all_thresholds)
            max_threshold = np.max(all_thresholds)

            prob_masks_and_diffs = list(zip(prob_masks, diffs))

            @njit
            def bissection_search(
                prob_masks_and_diffs: list[tuple[np.ndarray, np.ndarray]],
                alphas: list[float],
                precision: float = 1e-10,
            ):
                selected_thresholds = Dict.empty(
                    key_type=types.float64, value_type=types.float64
                )
                for alpha in alphas:
                    selected_thresholds[alpha] = np.inf

                def compute_conformal_risk(threshold):
                    risk_sum = 0.0
                    for prob_mask, diff in prob_masks_and_diffs:
                        this_risk = 0.0
                        for i in range(prob_mask.shape[0]):
                            for j in range(prob_mask.shape[1]):
                                if prob_mask[i, j] <= threshold:
                                    this_risk = max(this_risk, diff[i, j])
                        risk_sum = risk_sum + this_risk
                    return risk_sum / (n + 1) + 3 / (n + 1)

                for alpha in alphas:
                    lower = min_threshold
                    upper = max_threshold
                    best_threshold = -np.inf

                    risk_lower = compute_conformal_risk(lower)
                    risk_upper = compute_conformal_risk(upper)
                    assert risk_lower <= risk_upper

                    if compute_conformal_risk(upper) <= alpha:
                        # In this case we're already done
                        selected_thresholds[alpha] = upper
                    else:
                        while upper - lower > precision:
                            # assert risk_lower <= alpha <= risk_upper
                            middle = (lower + upper) / 2
                            assert lower <= middle <= upper
                            risk_middle = compute_conformal_risk(middle)
                            if risk_middle <= alpha:
                                best_threshold = middle
                                lower = middle
                                risk_lower = risk_middle
                            else:
                                upper = middle
                                risk_upper = risk_middle
                        selected_thresholds[alpha] = best_threshold

                return selected_thresholds

            print("before")
            before = time.time()
            selected_thresholds = bissection_search(prob_masks_and_diffs, alphas)
            after = time.time()
            print(f"after: {after - before}")

            return cls(
                base_model=base_model,
                thresholds={float(k): float(v) for k, v in selected_thresholds.items()},
            )

    def predict(
        self, low_resolution_image: Float[np.ndarray, "w h c"], *, alpha: float
    ) -> tuple[Float[np.ndarray, "w2 h2 c"], Bool[np.ndarray, "w2 h2"]]:
        assert (
            alpha in self.thresholds
        ), f"model not calibrated for fidelity level {alpha}"
        assert self.base_model is not None, "no base_model to predict with"
        high_resolution_image, mask_scores = self.base_model.predict(
            low_resolution_image
        )
        return high_resolution_image, mask_scores >= self.thresholds[alpha]
