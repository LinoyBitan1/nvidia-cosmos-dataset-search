# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for LinearClassifier model."""


import numpy as np
import pytest

from src.models.linear_classifier.model import LinearClassifier


@pytest.mark.parametrize("n_features", [2, 256])
@pytest.mark.parametrize("n_samples", [10, 20])
def test_model_train(n_features: int, n_samples: int) -> None:
    """Test the `train` method of the `LinearClassifier` class."""

    clf = LinearClassifier()

    inputs = np.random.rand(n_samples, n_features)
    targets = np.empty(n_samples)
    targets[: (n_samples // 2)] = True
    targets[(n_samples // 2) :] = False

    clf.train(inputs=inputs, targets=targets)
    coef = clf.grid_search.best_estimator_.coef_
    intercept = clf.grid_search.best_estimator_.intercept_

    assert coef.shape == (1, n_features)
    assert intercept.shape == (1,)
    assert not np.isnan(coef).any()
    assert not np.isnan(intercept).any()
