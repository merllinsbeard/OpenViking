# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Real SDK add methods with explicitly MOCK HTTP transport."""

import json
from pathlib import Path

import httpx
import pytest
from openviking_sdk import (
    AcceptedOrigin,
    ActorClaim,
    ActorRef,
    AsyncHTTPClient,
    ContributorStatement,
    FieldBasis,
    OriginContractMismatch,
    OriginStatements,
    SourceLocator,
    SourceRef,
    SyncHTTPClient,
    TimeStatement,
)


def contract():
    return json.loads(
        (Path(__file__).parents[3] / "tests/fixtures/message-origin-v1.json").read_text()
    )


def origin():
    return OriginStatements(
        contributors=(
            ContributorStatement(
                locator=SourceLocator("synthetic", "session-A"),
                refs=(SourceRef("provider_record", "record-A"),),
                times=(
                    TimeStatement(
                        "2026-10-17T09:10:11.123456+07:00",
                        "rfc3339",
                        "source_event",
                        FieldBasis("synthetic", "event.time"),
                    ),
                ),
                transport_sender=ActorClaim(
                    ActorRef("sender-A", "chat-A"), FieldBasis("synthetic", "event.sender")
                ),
            ),
        )
    )


def options():
    message = contract()["message"]
    return {
        key: value for key, value in message.items() if key not in ("role", "content", "origin")
    } | {"origin": origin()}


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
async def test_origin_add_methods_check_readback_and_return_domain_facts(batch):
    fixture = contract()
    requests = []

    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body == {"messages": [fixture["message"]]} if batch else body == fixture["message"]
        return httpx.Response(200, json=fixture["response"])

    client = AsyncHTTPClient(url="https://synthetic.invalid")
    client._http = httpx.AsyncClient(
        base_url="https://synthetic.invalid", transport=httpx.MockTransport(transport)
    )
    try:
        result = (
            await client.batch_add_messages(
                "native-v1",
                [{"role": "user", "content": fixture["message"]["content"], **options()}],
            )
            if batch
            else await client.add_message(
                "native-v1", "user", fixture["message"]["content"], options=options()
            )
        )
        assert len(requests) == 1
        accepted = result["accepted_origins"][0]
        assert isinstance(accepted, AcceptedOrigin)
        assert accepted.message_id == "msg-A"
        assert accepted.origin.statements == origin()
        assert accepted.origin.receiver.session_ingest_started_at == "2026-10-17T00:00:00Z"
        assert accepted.legacy_created_at == "2026-10-17T09:10:11.123456+07:00"
        assert result["message_count"] == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["missing", "partial", "literal", "version", "duplicate", "timeout", "strict"]
)
async def test_origin_mismatch_or_uncertain_append_is_terminal(failure):
    response = contract()["response"]
    member = response["result"]["accepted_origins"][0]
    if failure == "missing":
        response["result"].pop("accepted_origins")
    elif failure == "partial":
        member.pop("source_message_ids")
    elif failure == "literal":
        member["origin"]["contributors"][0]["times"][0]["literal"] = "2026-10-17T02:10:11.123456Z"
    elif failure == "version":
        member["origin"]["version"] = 2
    elif failure == "duplicate":
        response["result"]["accepted_origins"].append(member)
    calls = []

    def transport(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic ACK loss", request=request)
        return httpx.Response(422 if failure == "strict" else 200, json=response)

    client = AsyncHTTPClient(url="https://synthetic.invalid")
    client._http = httpx.AsyncClient(
        base_url="https://synthetic.invalid", transport=httpx.MockTransport(transport)
    )
    try:
        with pytest.raises(OriginContractMismatch):
            await client.add_message(
                "native-v1", "user", "Synthetic forwarded text.", options=options()
            )
        assert len(calls) == 1
    finally:
        await client.close()


def test_sync_facade_uses_the_same_origin_gate():
    client = SyncHTTPClient(url="https://synthetic.invalid")
    client._async_client._http = httpx.AsyncClient(
        base_url="https://synthetic.invalid",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"status": "ok", "result": {"message_count": 1}}
            )
        ),
    )
    try:
        with pytest.raises(OriginContractMismatch):
            client.add_message("native-v1", "user", "synthetic", options=options())
    finally:
        client.close()
