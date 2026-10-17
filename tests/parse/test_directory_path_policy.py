# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json
import posixpath
import re
from pathlib import Path

import pytest

from openviking.parse import registry
from openviking.parse.feishu_import import FeishuImportPlan
from openviking.parse.output import LocalParseOutputStore, read_artifact_manifest
from openviking.parse.parsers.directory import DirectoryParser
from openviking.server.identity import RequestContext, Role
from openviking.service.resource_service import ResourceService
from openviking.storage.queuefs.add_resource_msg import AddResourceMsg
from openviking_cli.exceptions import InvalidArgumentError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.open_viking_config import OpenVikingConfigSingleton


@pytest.fixture
def parser_config(monkeypatch):
    previous = OpenVikingConfigSingleton._instance
    OpenVikingConfigSingleton.initialize(config_dict={})
    monkeypatch.setattr(registry, "_default_registry", None)
    yield
    OpenVikingConfigSingleton._instance = previous


def write_sources(root: Path, sources: dict[str, str]) -> None:
    for relative, content in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


async def parse_files(root: Path, store: LocalParseOutputStore, **options) -> dict[str, bytes]:
    result = await DirectoryParser().parse(root, parse_output_store=store, **options)
    assert result.meta["failed_files"] == [], result.warnings
    ref = result.artifact_ref
    assert ref is not None
    manifest = await read_artifact_manifest(store, ref)
    prefix = ref.resource_rel + "/"
    files = {
        relative.removeprefix(prefix): (Path(ref.root) / relative).read_bytes()
        for relative in manifest
    }
    await store.cleanup(ref)
    return files


def configure_pdf_understanding(**directory_limits) -> None:
    OpenVikingConfigSingleton.initialize(
        config_dict={
            "parser_api": {
                "enable": True,
                "extensions": ["pdf"],
                "host": "http://127.0.0.1:1",
                "api_key": "synthetic-noncredential",
            },
            "directory": directory_limits,
        }
    )
    registry._default_registry = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"parse_mode_overrides": {"journal": "no_split"}},
        {"split_content": False, "parse_mode_overrides": {}},
    ],
)
async def test_unmaterialized_feishu_no_split_rejected_before_output(
    tmp_path, parser_config, options
):
    root = tmp_path / "source"
    write_sources(root, {"local.md": "# Local\n\nNative body.\n"})
    placeholder = root / "journal/doc.md"
    placeholder.parent.mkdir()
    plan = FeishuImportPlan(root, use_understanding=True)
    plan.add(placeholder, "https://example.invalid/docx/synthetic", "synthetic-token", "url")
    artifacts = tmp_path / "artifacts"
    store = LocalParseOutputStore(str(artifacts))
    with pytest.raises(InvalidArgumentError, match="no_split.*materialized.*journal/doc.md"):
        await DirectoryParser().parse(
            root,
            parse_output_store=store,
            parser_backend="understanding",
            _feishu_import_plan=plan,
            **options,
        )
    assert not placeholder.exists()
    assert list(artifacts.rglob("*")) == []
    assert (root / "local.md").read_bytes() == b"# Local\n\nNative body.\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("split", [True, False])
@pytest.mark.parametrize("policy", [None, {"journal": "no_split"}])
async def test_materialized_feishu_markdown_retains_native_modes(
    tmp_path, parser_config, split, policy
):
    root = tmp_path / "source"
    write_sources(root, {"journal/doc.md": "# Document\n\nMaterialized body.\n"})
    plan = FeishuImportPlan(root, use_understanding=True)
    plan.add(root / "journal/doc.md", "https://example.invalid/docx/synthetic", "token", "markdown")
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    files = await parse_files(
        root,
        store,
        split_content=split,
        parser_backend="understanding",
        _feishu_import_plan=plan,
        **({"parse_mode_overrides": policy} if policy is not None else {}),
    )
    expected_path = "journal/doc/doc.md" if split and policy is None else "journal/doc.md"
    assert files == {expected_path: b"# Document\n\nMaterialized body.\n"}


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["max_depth", "max_files"])
@pytest.mark.parametrize("policy", [None, {}, {"missing": "default"}, {"a": "default"}])
async def test_native_only_policy_preserves_no_split_admission(
    tmp_path, parser_config, limit, policy
):
    configure_pdf_understanding(**{limit: 2 if limit == "max_depth" else 1})
    root = tmp_path / "source"
    sources = (
        {"a/b/c/note.md": "# Note\n\nNative body.\n"}
        if limit == "max_depth"
        else {"a/note.md": "# Note\n\nNative body.\n", "other.md": "# Other\n\nNative other.\n"}
    )
    write_sources(root, sources)
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    files = await parse_files(
        root,
        store,
        split_content=False,
        **({"parse_mode_overrides": policy} if policy is not None else {}),
    )
    if limit == "max_depth":
        expected = (
            {"a/b/c/note/note.md": b"# Note\n\nNative body.\n"}
            if policy == {"a": "default"}
            else {"a/b/c/note.md": b"# Note\n\nNative body.\n"}
        )
    else:
        expected = {
            "a/note/note.md"
            if policy == {"a": "default"}
            else "a/note.md": b"# Note\n\nNative body.\n",
            "other.md": b"# Other\n\nNative other.\n",
        }
    assert files == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["max_depth", "max_files"])
@pytest.mark.parametrize("policy", [False, True])
async def test_understanding_candidate_retains_native_scan_limits(
    tmp_path, parser_config, limit, policy
):
    configure_pdf_understanding(**{limit: 2 if limit == "max_depth" else 1})
    root = tmp_path / "source"
    write_sources(
        root,
        {"a/b/c/paper.pdf": "%PDF-1.7\n"}
        if limit == "max_depth"
        else {"a.md": "# A\n\nNative body.\n", "paper.pdf": "%PDF-1.7\n"},
    )
    artifacts = tmp_path / "artifacts"
    store = LocalParseOutputStore(str(artifacts))
    expected = (
        "depth=3, max_depth=2, path=a/b/c" if limit == "max_depth" else "count=2, max_files=1"
    )
    with pytest.raises(InvalidArgumentError, match=expected):
        await DirectoryParser().parse(
            root,
            parse_output_store=store,
            split_content=not policy,
            **(
                {"parse_mode_overrides": {"paper.pdf": "default", "a": "default"}} if policy else {}
            ),
        )
    assert list(artifacts.rglob("*")) == []
    assert next(root.rglob("*.pdf")).read_bytes() == b"%PDF-1.7\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", [{"include": "*.md"}, {"exclude": "paper.pdf"}])
async def test_filtered_understanding_candidate_does_not_enable_limits(
    tmp_path, parser_config, selection
):
    configure_pdf_understanding(max_depth=1, max_files=1)
    root = tmp_path / "source"
    write_sources(
        root,
        {
            "a/b/paper.pdf": "%PDF-1.7\n",
            "a/b/note.md": "# Note\n\nNative body.\n",
            "other.md": "# Other\n\nNative other.\n",
        },
    )
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    files = await parse_files(
        root,
        store,
        split_content=False,
        parse_mode_overrides={"a/b/paper.pdf": "default"},
        **selection,
    )
    assert files == {
        "a/b/note.md": b"# Note\n\nNative body.\n",
        "other.md": b"# Other\n\nNative other.\n",
    }


@pytest.mark.asyncio
async def test_mixed_layout_preserves_two_import_paths(tmp_path, parser_config):
    sources = {
        "journal/day.md": "# Day\n\nDaily body.\n",
        "journal/nested/note.md": "# Nested\n\nNested body.\n",
        "docs/guide.md": "# Guide\n\nGuide body.\n",
        "journal-other/page.md": "# Other\n\nOther body.\n",
    }
    root = tmp_path / "source"
    write_sources(root, sources)
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    daily = await parse_files(root / "journal", store, split_content=False)
    other = await parse_files(root, store, ignore_dirs={"journal"})
    previous = {**{f"journal/{path}": body for path, body in daily.items()}, **other}
    assert previous == {
        "journal/day.md": sources["journal/day.md"].encode(),
        "journal/nested/note.md": sources["journal/nested/note.md"].encode(),
        "docs/guide/guide.md": sources["docs/guide.md"].encode(),
        "journal-other/page/page.md": sources["journal-other/page.md"].encode(),
    }
    global_default = await parse_files(root, store)
    global_no_split = await parse_files(root, store, split_content=False)
    assert set(global_default) == {
        "journal/day/day.md",
        "journal/nested/note/note.md",
        "docs/guide/guide.md",
        "journal-other/page/page.md",
    }
    assert set(global_no_split) == {
        "journal/day.md",
        "journal/nested/note.md",
        "docs/guide.md",
        "journal-other/page.md",
    }
    assert global_default != previous
    assert global_no_split != previous
    mixed = await parse_files(root, store, parse_mode_overrides={"journal": "no_split"})
    assert mixed == previous
    assert {path: (root / path).read_text(encoding="utf-8") for path in sources} == sources


@pytest.mark.asyncio
async def test_nested_overrides_and_flatten(tmp_path, parser_config):
    sources = {
        "journal/day.md": "# Day\n\nBody.\n",
        "journal/docs/guide.md": "# Guide\n\nBody.\n",
        "journal/docs/raw/note.md": "# Note\n\nBody.\n",
        "journal-other/page.md": "# Page\n\nBody.\n",
    }
    root = tmp_path / "source"
    write_sources(root, sources)
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    policy = {"journal/docs/raw": "no_split", "journal": "no_split", "journal/docs": "default"}
    files = await parse_files(root, store, parse_mode_overrides=policy)
    assert files == {
        "journal/day.md": sources["journal/day.md"].encode(),
        "journal/docs/guide/guide.md": sources["journal/docs/guide.md"].encode(),
        "journal/docs/raw/note.md": sources["journal/docs/raw/note.md"].encode(),
        "journal-other/page/page.md": sources["journal-other/page.md"].encode(),
    }
    flat = await parse_files(root, store, preserve_structure=False, parse_mode_overrides=policy)
    assert set(flat) == {"day/day.md", "guide/guide.md", "note/note.md", "page/page.md"}
    reverse = await parse_files(
        root, store, split_content=False, parse_mode_overrides={"journal/docs": "default"}
    )
    assert set(reverse) == {
        "journal/day.md",
        "journal/docs/guide/guide.md",
        "journal/docs/raw/note/note.md",
        "journal-other/page.md",
    }
    exact = await parse_files(
        root, store, parse_mode_overrides={"journal": "no_split", "journal/day.md": "default"}
    )
    assert set(exact) == {
        "journal/day/day.md",
        "journal/docs/guide.md",
        "journal/docs/raw/note.md",
        "journal-other/page/page.md",
    }


@pytest.mark.asyncio
async def test_cross_links_follow_target_policy(tmp_path, parser_config):
    sources = {
        "journal/day.md": "# Day\n\n[guide](../docs/guide.md#guide) [note](nested/note.md#note)\n",
        "journal/nested/note.md": "# Note\n\n[day](../day.md#day)\n",
        "docs/guide.md": "# Guide\n\n[day](../journal/day.md#day) [other](other.md#other)\n",
        "docs/other.md": "# Other\n\n[guide](guide.md#guide)\n",
    }
    root = tmp_path / "source"
    write_sources(root, sources)
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    files = await parse_files(root, store, parse_mode_overrides={"journal": "no_split"})
    assert files == {
        "journal/day.md": b"# Day\n\n[guide](../docs/guide/guide.md#guide) [note](nested/note.md#note)\n",
        "journal/nested/note.md": b"# Note\n\n[day](../day.md#day)\n",
        "docs/guide/guide.md": b"# Guide\n\n[day](../../journal/day.md#day) [other](../other/other.md#other)\n",
        "docs/other/other.md": b"# Other\n\n[guide](../guide/guide.md#guide)\n",
    }
    for path, body in files.items():
        for link in re.findall(r"\]\(([^)]+)\)", body.decode()):
            target = link.split("#")[0]
            assert posixpath.normpath(posixpath.join(posixpath.dirname(path), target)) in files
    assert {path: (root / path).read_text(encoding="utf-8") for path in sources} == sources


@pytest.mark.asyncio
async def test_long_heading_targets_and_frontmatter(tmp_path, parser_config):
    alpha = "# Alpha\n\n" + "Alpha paragraph. " * 240 + "\n\n"
    beta = "# Beta\n\n" + "Beta paragraph. " * 240 + "\n"
    sources = {
        "docs/long.md": alpha + beta,
        "journal/day.md": "# Day\n\n[alpha](../docs/long.md#alpha) [beta](../docs/long.md#beta)\n",
        "journal/Юникод note.md": (
            "---\ntitle: Журнал дня\n---\n# Заголовок\n\nТело. [guide](../docs/space name.md#guide)\n"
        ),
        "docs/space name.md": (
            "---\ntitle: Guide title\n---\n# Guide\n\nBody. [day](../journal/Юникод note.md#заголовок)\n"
        ),
    }
    root = tmp_path / "source"
    write_sources(root, sources)
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    files = await parse_files(root, store, parse_mode_overrides={"journal": "no_split"})
    assert set(files) == {
        "docs/long/Alpha.md",
        "docs/long/Beta.md",
        "journal/day.md",
        "journal/Журнал_дня.md",
        "docs/Guide_title/Guide_title.md",
    }
    assert files["journal/day.md"] == (
        b"# Day\n\n[alpha](../docs/long/Alpha.md#alpha) [beta](../docs/long/Beta.md#beta)\n"
    )
    assert files["docs/long/Alpha.md"].decode().strip() == alpha.strip()
    assert files["docs/long/Beta.md"].decode().strip() == beta.strip()
    assert files["journal/Журнал_дня.md"].decode() == (
        "---\ntitle: Журнал дня\n---\n# Заголовок\n\nТело. "
        "[guide](../docs/Guide_title/Guide_title.md#guide)\n"
    )
    assert files["docs/Guide_title/Guide_title.md"].decode() == (
        "---\ntitle: Guide title\n---\n# Guide\n\nBody. "
        "[day](../../journal/Журнал_дня.md#заголовок)\n"
    )
    no_split = await parse_files(root, store, split_content=False)
    assert no_split["docs/long.md"] == (alpha + beta).encode()


@pytest.mark.asyncio
async def test_no_option_and_native_link_gaps(tmp_path, parser_config):
    root = tmp_path / "source"
    write_sources(
        root,
        {
            "journal/day.md": "# Day\n\n[missing](missing.md#absent) [ref][g]\n\n[g]: ../docs/guide.md#guide\n",
            "docs/guide.md": "# Guide\n\n[day](../journal/day.md#day)\n",
        },
    )
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    for split in (True, False):
        legacy = await parse_files(root, store, split_content=split)
        empty_policy = await parse_files(root, store, split_content=split, parse_mode_overrides={})
        assert legacy == empty_policy
    files = await parse_files(root, store, parse_mode_overrides={"journal": "no_split"})
    assert files["journal/day.md"] == (
        b"# Day\n\n[missing](missing.md#absent) [ref][g]\n\n[g]: ../docs/guide.md#guide\n"
    )
    assert files["docs/guide/guide.md"] == b"# Guide\n\n[day](../../journal/day.md#day)\n"


@pytest.mark.asyncio
async def test_native_queue_args_round_trip(tmp_path, parser_config):
    root = tmp_path / "source"
    write_sources(
        root, {"journal/day.md": "# Day\n\nBody.\n", "docs/guide.md": "# Guide\n\nBody.\n"}
    )
    ctx = RequestContext(user=UserIdentifier("fixture", "reader"), role=Role.USER)
    service = ResourceService()
    args = {"parse_mode": "no_split", "parse_mode_overrides": {"docs": "default"}}
    first = await service._normalize_add_resource_args(args, watch_interval=0, ctx=ctx)
    msg = AddResourceMsg(
        task_id="fixture-task",
        root_uri="viking://resources/fixture",
        account_id="fixture",
        user_id="reader",
        role=Role.USER,
        path=str(root),
        args=first.processor_kwargs,
        parse_mode=first.parse_mode.value,
    )
    wire = json.loads(json.dumps(msg.to_dict()))
    assert wire["args"]["parse_mode_overrides"] == {"docs": "default"}
    replay = AddResourceMsg.from_dict(wire)
    second = await service._normalize_add_resource_args(replay.args, watch_interval=0, ctx=ctx)
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    files = await parse_files(
        root, store, split_content=replay.parse_mode == "default", **second.processor_kwargs
    )
    assert files == {
        "journal/day.md": b"# Day\n\nBody.\n",
        "docs/guide/guide.md": b"# Guide\n\nBody.\n",
    }


@pytest.mark.asyncio
async def test_policy_does_not_expand_source_selection(tmp_path, parser_config):
    root = tmp_path / "source"
    write_sources(
        root,
        {
            "journal/day.md": "# Day\n\nBody.\n",
            "journal/ignored.md": "# Ignored\n",
            "journal/.hidden.md": "# Hidden\n",
            "docs/guide.md": "# Guide\n\nBody.\n",
            ".gitignore": "ignored.md\n",
            "journal/attachment.bin": "attachment",
        },
    )
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    files = await parse_files(
        root, store, include="*.md", parse_mode_overrides={"journal": "no_split"}
    )
    assert files == {
        "journal/day.md": b"# Day\n\nBody.\n",
        "docs/guide/guide.md": b"# Guide\n\nBody.\n",
    }
    filtered = await parse_files(
        root,
        store,
        include="*.md",
        exclude="guide.md",
        parse_mode_overrides={"journal": "no_split"},
    )
    assert filtered == {"journal/day.md": b"# Day\n\nBody.\n"}


@pytest.mark.asyncio
async def test_case_collision_uses_native_failure_reporting(tmp_path, parser_config):
    root = tmp_path / "source"
    write_sources(
        root,
        {
            "journal/a.md": "---\ntitle: Same\n---\n# First\n",
            "journal/b.md": "---\ntitle: same\n---\n# Second\n",
        },
    )
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    result = await DirectoryParser().parse(
        root, parse_output_store=store, parse_mode_overrides={"journal": "no_split"}
    )
    assert result.meta["file_count"] == 1
    assert result.meta["failed_files"][0]["path"] == "journal/b.md"
    assert "case-only name conflict" in result.meta["failed_files"][0]["error"]
    ref = result.artifact_ref
    assert ref is not None
    assert (Path(ref.root) / ref.resource_rel / "journal/Same.md").read_bytes() == (
        b"---\ntitle: Same\n---\n# First\n"
    )
    await store.cleanup(ref)


@pytest.mark.asyncio
async def test_exact_title_collision_remains_a_native_gap(tmp_path, parser_config):
    root = tmp_path / "source"
    write_sources(
        root,
        {
            "journal/a.md": "---\ntitle: Same\n---\n# First\n",
            "journal/b.md": "---\ntitle: Same\n---\n# Second\n",
        },
    )
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    files = await parse_files(root, store, parse_mode_overrides={"journal": "no_split"})
    assert files == {"journal/Same.md": b"---\ntitle: Same\n---\n# Second\n"}
    assert (root / "journal/a.md").read_bytes() == b"---\ntitle: Same\n---\n# First\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        None,
        [],
        "journal:no_split",
        {"": "no_split"},
        {"/journal": "no_split"},
        {"../journal": "no_split"},
        {"journal/../docs": "no_split"},
        {"./journal": "no_split"},
        {"journal//nested": "no_split"},
        {"journal/": "no_split"},
        {"C:/journal": "no_split"},
        {"C:journal": "no_split"},
        {"journal\\nested": "no_split"},
        {"journal\x00": "no_split"},
        {1: "no_split"},
        {"journal": "raw"},
        {"journal": False},
        {"journal": None},
        {"journal": ["no_split"]},
    ],
)
async def test_policy_rejected_at_external_boundaries(tmp_path, parser_config, overrides):
    ctx = RequestContext(user=UserIdentifier("fixture", "reader"), role=Role.USER)
    with pytest.raises(InvalidArgumentError, match="parse_mode_overrides"):
        await ResourceService()._normalize_add_resource_args(
            {"parse_mode_overrides": overrides}, watch_interval=0, ctx=ctx
        )
    root = tmp_path / "source"
    write_sources(root, {"journal/day.md": "# Day\n\nBody.\n"})
    store = LocalParseOutputStore(str(tmp_path / "artifacts"))
    with pytest.raises(InvalidArgumentError, match="parse_mode_overrides"):
        await DirectoryParser().parse(
            root, parse_output_store=store, parse_mode_overrides=overrides
        )
    assert list((tmp_path / "artifacts").iterdir()) == []
