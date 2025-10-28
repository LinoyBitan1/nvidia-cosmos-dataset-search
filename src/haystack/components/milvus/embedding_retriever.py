# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Embedding retriever Haystack 2.x implementation for Milvus.

Taken from:
https://gitlab-master.nvidia.com/dl/ai-services/microservices/retrieval/-/blob/main/src/retrieval/components/milvus/milvus.py
"""

from typing import Any, Dict, List, Optional

import numpy as np
from haystack import component
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import Document

from src.haystack.components.milvus.document_store import MilvusDocumentStore


@component
class MilvusEmbeddingRetriever:
    """
    Uses a vector similarity metric to retrieve documents from the MilvusDocumentStore.

    Needs to be connected to the MilvusDocumentStore to run.
    """

    def __init__(
        self,
        *,
        document_store: MilvusDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        reconstruct: bool = False,
    ):
        """
        Create the MilvusEmbeddingRetriever component.

        :param document_store: An instance of MilvusDocumentStore.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
            Filters are applied during the approximate kNN search to ensure that top_k matching documents are returned.
        :param top_k: Maximum number of Documents to return, defaults to 10
        :param reconstruct: Whether to return the embedding of the retrieved documents. Defaults to False.
        :raises ValueError: If `document_store` is not an instance of MilvusDocumentStore.
        """
        if not isinstance(document_store, MilvusDocumentStore):
            msg = "document_store must be an instance of MilvusDocumentStore"
            raise ValueError(msg)

        top_k = int(top_k)
        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._reconstruct = reconstruct

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            filters=self._filters,
            top_k=self._top_k,
            reconstruct=self._reconstruct,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MilvusEmbeddingRetriever":
        init_parameters = data["init_parameters"]
        document_store = MilvusDocumentStore.from_dict(
            init_parameters["document_store"]
        )
        return default_from_dict(
            cls,
            {
                **data,
                "init_parameters": {
                    **init_parameters,
                    "document_store": document_store,
                },
            },
        )

    @component.output_types(documents=List[List[Document]])
    def run(
        self,
        query_embeddings: List[List[float]],
        index_name: str = "",
        top_k: Optional[int] = None,
        search_params: Dict = {},
        filters: Dict = {},
        reconstruct: Optional[bool] = None,
    ) -> Dict[str, List[List[Document]]]:
        """
        Retrieve documents using a vector similarity metric.

        :param index_name: collection name to query.
        :param query_embeddings: Batch of embeddings to query.
        :param top_k: number of nearest neighbors to return.
        :param search_params: search params specific to different types of indices, for example nprobe.
        :param filters: extra filters for the search.
        :param reconstruct: whether to return embeddings.
        :return: List of Document similar to each of `query_embeddings`.
        """
        docs = []
        # (TODO francesco: avoid loop)
        for embedding in query_embeddings:
            docs.append(
                self._document_store._embedding_retrieval(
                    collection_name=index_name,
                    query_embedding=np.array(embedding),
                    filters=filters,
                    search_params=search_params,
                    top_k=top_k or self._top_k,
                    return_embedding=reconstruct or self._reconstruct,
                )
            )
        return {"documents": docs}
