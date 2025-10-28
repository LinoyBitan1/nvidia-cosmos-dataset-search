# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Haystack components for document filtering models."""

import base64
import pickle
from typing import Any, Dict, List

import numpy as np
from haystack import Document, component

from src.haystack.serializer import SerializerMixin


@component
class LinearClassifierFilter(SerializerMixin):
    """Class definition for a linear classifier which is used to filter results based on the classification score.

    The classifier object is passed as base64 encoded string.
    """

    @component.output_types(documents=List[Document])
    def run(
        self,
        documents: List[Document],
        clf: Dict[str, Any] = {},
    ):
        if not clf:
            return {"documents": documents}

        if not documents:
            return {"documents": []}

        if documents[0].embedding is None:
            raise TypeError("To enable filtering, please set reconstuct=True")

        best_model = pickle.loads(base64.b64decode(clf["model"])).get_best_model()

        all_embeddings = np.array([doc.embedding for doc in documents])
        preds = best_model.predict(all_embeddings)

        filtered_documents = []
        for doc, pred in zip(documents, preds):
            if pred:
                doc.embedding = None
                filtered_documents.append(doc)

        return {"documents": filtered_documents}
