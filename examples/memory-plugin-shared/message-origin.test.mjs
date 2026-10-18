import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { sendSessionMessages } from "./lib/batch-send.mjs";
import { OriginContractMismatch, validateOriginReadback } from "./lib/message-origin.mjs";

const golden = JSON.parse(await readFile(new URL("../../tests/fixtures/message-origin-v1.json", import.meta.url)));

test("native sender verifies literal persistent origin before advancing", async () => {
  const observed = [];
  const sent = await sendSessionMessages(async (_path, init) => {
    const payload = JSON.parse(init.body);
    assert.deepEqual(payload.messages, [golden.message]);
    observed.push("readback");
    return { ok: true, status: 200, result: golden.response.result };
  }, "native-v1", [golden.message], { onSent: (count) => observed.push(count) });
  assert.deepEqual(observed, ["readback", 1]);
  assert.equal(sent.sent, 1);
});

for (const failure of ["omitted", "partial", "literal", "version", "extra", "strict", "lost"]) {
  test(`origin ${failure} cannot advance or enqueue a capture cursor`, async () => {
    const response = structuredClone(golden.response.result);
    if (failure === "omitted") delete response.accepted_origins;
    if (failure === "partial") delete response.accepted_origins[0].origin.receiver;
    if (failure === "literal") response.accepted_origins[0].origin.contributors[0].times[0].literal = "2026-10-17T02:10:11.123456Z";
    if (failure === "version") response.accepted_origins[0].origin.version = 2;
    if (failure === "extra") response.accepted_origins.push(structuredClone(response.accepted_origins[0]));
    let calls = 0;
    let cursor = 0;
    const result = await sendSessionMessages(async () => {
      calls++;
      if (failure === "lost") throw new Error("MOCK lost HTTP acknowledgement after append");
      return failure === "strict" ? { ok: false, status: 422 } : { ok: true, status: 200, result: response };
    }, "native-v1", [golden.message], { enqueueOnRetryable: true, onSent: (count) => { cursor += count; } });
    assert.equal(calls, 1);
    assert.equal(cursor, 0);
    assert.equal(result.sent, 0);
    assert.equal(result.queued, 0);
    assert.equal(result.retryable, false);
    assert.ok(result.lastError instanceof OriginContractMismatch);
  });
}

test("serial endpoint fallback must verify the same origin receipt", async () => {
  let calls = 0;
  const result = await sendSessionMessages(async () => {
    calls++;
    return calls === 1 ? { ok: false, status: 404 } : { ok: true, status: 200, result: { message_count: 1 } };
  }, "old-server", [golden.message]);
  assert.equal(calls, 2);
  assert.equal(result.sent, 0);
  assert.ok(result.lastError instanceof OriginContractMismatch);
});

test("readback requires complete fanout coverage and keeps shared candidates", () => {
  const message = { ...golden.message, message_kind: undefined, parts: [{ type: "tool" }, { type: "tool" }] };
  const response = { origin_readback: "jsonl", accepted_origins: [0, 1].map((index) => {
    const member = structuredClone(golden.response.result.accepted_origins[0]);
    member.message_id = `physical-${index}`;
    member.message_kind = "tool_transport";
    member.origin.receiver.applicability = { kind: "input_group", group_id: "group-A", member_index: index, member_count: 2 };
    return member;
  }) };
  validateOriginReadback([message], response);
  response.accepted_origins.pop();
  assert.throws(() => validateOriginReadback([message], response), OriginContractMismatch);
});
