# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Custom document writer for writing to multiple collections in a thread-safe manner."""

from typing import Any, Dict, List, Optional
import logging

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import DuplicatePolicy

from src.haystack.components.milvus.document_store import MilvusDocumentStore

logger = logging.getLogger(__name__)


@component
class MilvusDocumentWriter:
    def __init__(
        self,
        document_store: MilvusDocumentStore,
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
        index_name: str = "",
    ):
        self.document_store = document_store
        self.policy = policy
        self.index_name = index_name

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            policy=self.policy,
            index_name=self.index_name,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MilvusDocumentWriter":
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

    @component.output_types(documents_written=int)
    def run(
        self,
        documents: List[Document],
        policy: Optional[DuplicatePolicy] = None,
        index_name: str = "",
    ):
        if policy is None:
            policy = self.policy
        logger.debug("MilvusWriter received %d docs", len(documents))

        documents_written = self.document_store.write_documents(
            documents=documents,
            policy=policy,
            collection_name=index_name,
        )
        return {"documents_written": documents_written}
