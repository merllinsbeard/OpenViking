# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Backup behavior for user configuration and genuinely missing vectors."""

import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from openviking.server.identity import RequestContext, Role
from openviking.storage.ovpack.manifest import manifest_dense_info, read_manifest
from openviking.storage.ovpack.operations import backup_ovpack
from openviking.storage.ovpack.validation import base_name_from_entries, validate_manifest_content
from openviking_cli.exceptions import InvalidArgumentError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.embedding_config import EmbeddingConfig, EmbeddingModelConfig


class BackupFS:
    def __init__(self, extra_path=None):
        self.files = {
            f"viking://user/{user}/settings/{name}": b'{"test":true}'
            for user in ("alice", "bob")
            for name in ("user_config.json", "user_config.backup.json")
        }
        self.files["viking://user/alice/memories/example.txt"] = b"a user memory"
        if extra_path:
            self.files[extra_path] = b"a separate user file"

    async def tree(self, uri, **kwargs):
        if uri == "viking://resources":
            return []
        assert uri == "viking://user"
        prefix = uri + "/"
        return [
            {"uri": path, "rel_path": path[len(prefix):], "isDir": False, "size": len(data)}
            for path, data in self.files.items()
        ]

    async def exists(self, uri, ctx=None):
        return False

    async def read_file_bytes(self, uri, ctx=None):
        return self.files[uri]


class VectorStore:
    def __init__(self, missing=False):
        self.missing = missing

    async def resolve(self, account_id):
        return SimpleNamespace(
            embedding=EmbeddingConfig(
                dense=EmbeddingModelConfig(
                    provider="openai", model="test", dimension=3, api_key="test"
                )
            ),
            vectordb=SimpleNamespace(sparse_weight=0),
        )

    async def get_account_backend(self, account_id):
        return self

    async def get_collection_meta(self):
        return {}

    async def filter(self, **kwargs):
        uri = kwargs["filter"].value
        if not self.missing and uri == "viking://user/alice/memories/example.txt":
            return [{"uri": uri, "context_type": "memory", "level": 2, "vector": [0.1, 0.2, 0.3]}]
        return []


class BackupUserConfigTest(unittest.IsolatedAsyncioTestCase):
    async def backup(self, fs, path, store):
        return await backup_ovpack(
            fs,
            str(path),
            ctx=RequestContext(user=UserIdentifier("acct", "alice"), role=Role(Role.ADMIN)),
            vector_store=store,
            vector_config_resolver=store,
            include_vectors=True,
        )

    async def test_keeps_user_configs_in_vector_backup(self):
        fs = BackupFS()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backup.ovpack"
            await self.backup(fs, path, VectorStore())
            with zipfile.ZipFile(path) as archive:
                self.assertIsNone(archive.testzip())
                info = archive.infolist()
                manifest = read_manifest(archive, base_name_from_entries(info))
                validate_manifest_content(archive, manifest, info, "openviking-backup")
                dense = manifest_dense_info(manifest)
                assert dense is not None
                self.assertGreater(dense["count"], 0)
                for user in ("alice", "bob"):
                    for name in ("user_config.json", "user_config.backup.json"):
                        self.assertEqual(
                            archive.read(f"openviking-backup/files/user/{user}/settings/{name}"),
                            fs.files[f"viking://user/{user}/settings/{name}"],
                        )
                self.assertIn(
                    "openviking-backup/files/user/alice/memories/example.txt", archive.namelist()
                )

    async def test_rejects_missing_memory_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InvalidArgumentError) as error:
                await self.backup(
                    BackupFS(), Path(directory) / "backup.ovpack", VectorStore(missing=True)
                )
            self.assertEqual(error.exception.details["missing_record_count"], 1)
            self.assertIn("example.txt", error.exception.details["missing_records"][0])

    async def test_same_name_elsewhere_is_not_exempt(self):
        fs = BackupFS("viking://user/alice/memories/user_config.json")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InvalidArgumentError) as error:
                await self.backup(fs, Path(directory) / "backup.ovpack", VectorStore())
            self.assertEqual(error.exception.details["missing_record_count"], 1)
            self.assertIn(
                "memories/user_config.json", error.exception.details["missing_records"][0]
            )


if __name__ == "__main__":
    unittest.main()
