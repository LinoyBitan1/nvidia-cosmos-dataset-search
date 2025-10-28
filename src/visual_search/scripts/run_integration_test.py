# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Image ingestion script for visual search."""

import logging

from src.visual_search.client import Client
from src.visual_search.client.config import DEFAULT

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TestCases:
    def __init__(self, profile: str) -> None:
        self.profile = profile
        logging.info(f"Running tests with profile {profile}")
        self.client = Client()

    def run(self) -> None:
        """Run tests."""

        pipeline = "c_radio_image_search_milvus"
        collection_name = "Test Collection"

        collection_id = None
        exception = None

        try:
            logging.info("Listing all pipelines")
            pipelines = self.client.pipelines.list(profile=self.profile)
            ids = {p["id"] for p in pipelines["pipelines"]}
            assert {pipeline}.issubset(ids)

            logging.info("Creating collection")
            collection = self.client.collections.create(
                pipeline=pipeline, profile=self.profile, name=collection_name
            )
            assert collection["collection"]["pipeline"] == pipeline
            assert collection["collection"]["name"] == collection_name
            collection_id = collection["collection"]["id"]

            logging.info("Get collection")
            collection = self.client.collections.get(
                collection_id=collection_id, profile=self.profile
            )
            assert collection["collection"]["id"] == collection_id
            assert collection["collection"]["pipeline"] == pipeline
            assert collection["collection"]["name"] == collection_name
            assert collection["total_documents_count"] == 0

            logging.info("Listing all collections")
            collections = self.client.collections.list(profile=self.profile)
            ids = {collection["id"] for collection in collections["collections"]}
            assert {collection_id}.issubset(ids)

            logging.info("Searching on collection")
            results = self.client.search(
                collection_ids=[collection_id],
                text_query="picture of a car",
                top_k=2,
                profile=self.profile,
            )
            retrievals = results["retrievals"]
            assert len(retrievals) == 0
        except Exception as e:
            exception = e

        if collection_id is not None:
            logging.info("Deleting collection")
            delete_response = self.client.collections.delete(
                collection_id=collection_id, profile=self.profile
            )
            assert "deleted" in delete_response.get("message", "").lower()
            assert delete_response["id"] == collection_id

        if exception is not None:
            raise exception


if __name__ == "__main__":
    TestCases(profile=DEFAULT).run()
