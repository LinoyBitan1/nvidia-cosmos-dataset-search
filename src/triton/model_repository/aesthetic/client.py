# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Triton client for aesthetic model."""

import time

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


class TritonAestheticClient:
    def __init__(self, url: str = "localhost:8000", model_name: str = "clip") -> None:
        self.url = url
        self.model_name = model_name
        self.client = httpclient.InferenceServerClient(url=url)

    def encode_image(self, batch: np.ndarray) -> np.ndarray:
        """Encode image batch into aesthetic scores."""

        input_tensors = [
            httpclient.InferInput(
                "image", batch.shape, datatype=np_to_triton_dtype(batch.dtype)
            )
        ]
        input_tensors[0].set_data_from_numpy(batch)

        response = self.client.infer(
            model_name=self.model_name,
            inputs=input_tensors,
            outputs=[httpclient.InferRequestedOutput("aesthetic_score")],
        )
        return response.as_numpy("aesthetic_score")


def test() -> None:
    client = TritonAestheticClient(model_name="aesthetic")
    images = np.random.randint(0, 256, (1, 224, 224, 3)).astype(np.uint8)
    timings = []
    for i in range(30):
        start = time.time()
        aesthetic_scores = client.encode_image(images)
        print(aesthetic_scores)
        stop = time.time()
        if i > 3:
            timings.append(stop - start)
    print(f"Average time: {np.mean(timings)}")


if __name__ == "__main__":
    test()
