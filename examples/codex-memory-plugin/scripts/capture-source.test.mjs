import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { extractCaptureTranscript, extractCaptureTurns } from "./capture-utils.mjs";
import { catchUpTurns } from "./ov-session.mjs";
import { CAPTURE_FORMAT_VERSION, loadState, saveState } from "./session-state.mjs";

const cfg = { captureAssistantTurns: true };
const source = {
  created_at: "2026-10-17T23:48:12.123456+05:30",
  turn_id: "turn-source",
  message_kind: "user_query",
  source_message_ids: ["source-a", "source-b"],
};

function message(text, metadata = {}) {
  return {
    timestamp: source.created_at,
    type: "response_item",
    payload: {
      type: "message", role: "user", content: [{ type: "input_text", text }],
      ...metadata,
    },
  };
}

test("keeps supplied source fields and the original offset on a selected message", () => {
  const { turns, excludedLegacyTurnIndices } = extractCaptureTranscript([
    message("Remember the source fields", { ...source, id: "native-message" }),
  ], cfg);
  assert.deepEqual(turns, [{
    role: "user", text: "Remember the source fields",
    parts: [{ type: "text", text: "Remember the source fields" }],
    ...source, source_message_ids: ["source-a", "source-b", "native-message"],
  }]);
  assert.deepEqual(excludedLegacyTurnIndices, []);
});

test("does not derive metadata from role, call_id, ordinal, context or capture time", () => {
  const turns = extractCaptureTurns([
    { type: "turn_context", payload: { turn_id: "unrelated-context" } },
    { ordinal: 42, ...message("Unknown source metadata"), timestamp: undefined },
    { type: "response_item", payload: { type: "function_call_output", call_id: "call-only", output: "Tool result" } },
    message("Known record timestamp", { id: "native-known" }),
  ], cfg);
  assert.deepEqual(turns.slice(0, 2), [
    { role: "user", text: "Unknown source metadata", parts: [{ type: "text", text: "Unknown source metadata" }] },
    { role: "user", text: "[tool-result] Tool result", parts: [{
      type: "tool", tool_id: "call-only", tool_name: undefined, tool_status: "completed", tool_output: "Tool result",
    }] },
  ]);
  assert.equal(turns[2].created_at, source.created_at);
  assert.deepEqual(turns[2].source_message_ids, ["native-known"]);
});

test("keeps metadata through custom tools, agent messages and nested messages", () => {
  const payloads = [
    { type: "custom_tool_call", call_id: "custom", name: "exec", input: "pwd" },
    { type: "custom_tool_call_output", call_id: "custom", output: "project root" },
    { type: "tool_search_call", id: "search-item", arguments: { query: "files" } },
    { type: "tool_search_output", tools: [] },
    { type: "agent_message", author: "worker", recipient: "main", content: [{ type: "output_text", text: "Completed the source check" }] },
    { type: "sub_agent_activity", event_id: "activity", kind: "completed" },
    { message: { role: "user", content: "Remember the nested source", ...source } },
  ];
  const turns = extractCaptureTurns(payloads.map((payload) => ({
    type: "response_item", timestamp: source.created_at,
    payload: { ...source, ...payload },
  })), cfg);
  assert.equal(turns.length, 7);
  for (const turn of turns) {
    assert.equal(turn.created_at, source.created_at);
    assert.equal(turn.turn_id, "turn-source");
    assert.equal(turn.message_kind, "user_query");
    assert.deepEqual(turn.source_message_ids.slice(0, 2), ["source-a", "source-b"]);
  }
  assert.equal(turns[1].parts[0].tool_output, "project root");
  assert.equal(turns[6].text, "Remember the nested source");
});

test("keeps child metadata and inherited turn identity through history append", () => {
  const turns = extractCaptureTurns([{
    timestamp: "2026-10-18T00:00:00Z", type: "history_mutation",
    payload: { operation: "append", turn_id: "history-turn", items: [
      { type: "message", role: "user", content: [{ type: "input_text", text: "History question" }],
        id: "history-message", created_at: source.created_at, source_message_ids: ["history-source"], message_kind: "checkpoint" },
    ] },
  }], cfg);
  assert.deepEqual(turns, [{
    role: "user", text: "History question", parts: [{ type: "text", text: "History question" }],
    created_at: source.created_at, turn_id: "history-turn", message_kind: "checkpoint",
    source_message_ids: ["history-source", "history-message"],
  }]);
});

test("keeps all contributing refs without choosing a conflicting tool timestamp", () => {
  const turns = extractCaptureTurns([
    { timestamp: source.created_at, type: "response_item", payload: {
      type: "function_call", call_id: "logical-tool", id: "call-item", name: "read_file", arguments: { path: "file.txt" },
      turn_id: "turn-tool", source_message_ids: ["source-call"], message_kind: "tool_transport",
    } },
    { timestamp: "2026-10-18T00:00:00Z", type: "response_item", payload: {
      type: "function_call_output", call_id: "logical-tool", id: "result-item", output: "old result",
      turn_id: "turn-tool", source_message_ids: ["source-result"], message_kind: "tool_transport",
    } },
    { timestamp: "2026-10-18T00:00:01Z", type: "event_msg", payload: {
      type: "item_completed", turn_id: "turn-tool", item: {
        type: "McpToolCall", id: "logical-tool", tool: "read_file", arguments: { path: "file.txt" },
        result: { content: [{ type: "text", text: "final result" }] }, status: "completed",
        source_message_ids: ["source-completed"], message_kind: "tool_transport",
      },
    } },
  ], cfg);
  assert.equal(turns.length, 2);
  assert.deepEqual(turns.map((turn) => turn.parts[0].tool_status), ["running", "completed"]);
  assert.equal(turns[1].parts[0].tool_output, "final result");
  for (const turn of turns) {
    assert.equal(turn.created_at, undefined);
    assert.equal(turn.turn_id, "turn-tool");
    assert.equal(turn.message_kind, "tool_transport");
    assert.deepEqual(new Set(turn.source_message_ids), new Set([
      "source-call", "call-item", "source-result", "result-item", "source-completed", "logical-tool",
    ]));
  }
});

test("response replay keeps the old first-wins content and unions refs", () => {
  const first = { timestamp: source.created_at, type: "response_item", payload: {
    type: "custom_tool_call", id: "item-first", call_id: "same-call", name: "exec", input: "pwd", source_message_ids: ["ref-first"],
  } };
  const replay = { ...first, payload: { ...first.payload, id: "item-replay", input: "different input", source_message_ids: ["ref-replay"] } };
  const turns = extractCaptureTurns([first, replay, replay], cfg);
  assert.equal(turns.length, 1);
  assert.deepEqual(turns[0].parts[0].tool_input, { value: "pwd" });
  assert.equal(turns[0].created_at, source.created_at);
  assert.deepEqual(turns[0].source_message_ids, ["ref-first", "item-first", "ref-replay", "item-replay"]);
  assert.deepEqual(first.payload.source_message_ids, ["ref-first"]);
});

test("payload builder preserves source fields in batch and serial fallback without replay", async (t) => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-codex-source-"));
  const previous = process.env.OPENVIKING_CODEX_STATE_DIR;
  process.env.OPENVIKING_CODEX_STATE_DIR = stateDir;
  t.after(async () => {
    if (previous === undefined) delete process.env.OPENVIKING_CODEX_STATE_DIR;
    else process.env.OPENVIKING_CODEX_STATE_DIR = previous;
    await rm(stateDir, { recursive: true, force: true });
  });
  for (const serial of [false, true]) {
    const id = `source-${serial}`;
    const transcriptPath = join(stateDir, `${id}.jsonl`);
    const records = [message("Captured prefix"), message("New source message", source)];
    await writeFile(transcriptPath, records.map(JSON.stringify).join("\n"));
    const state = await loadState(id);
    state.captureFormatVersion = CAPTURE_FORMAT_VERSION;
    state.capturedTurnCount = 1;
    await saveState(state);
    const sent = [];
    const fetchJSONRes = async (path, init) => {
      const body = JSON.parse(init.body);
      if (path.endsWith("/batch") && serial) return { ok: false, status: 404 };
      sent.push(...(body.messages ?? [body]));
      return { ok: true, status: 200 };
    };
    const first = await catchUpTurns({ state, transcriptPath, cfg, activePeerId: "project", fetchJSONRes });
    assert.equal(first.added, 1);
    assert.deepEqual(sent, [{ role: "user", parts: [{ type: "text", text: "New source message" }], peer_id: "project", ...source }]);
    assert.equal((await loadState(id)).capturedTurnCount, 2);
    assert.equal((await catchUpTurns({ state, transcriptPath, cfg, fetchJSONRes })).added, 0);
    assert.equal(sent.length, 1);
    assert.equal(JSON.parse(await readFile(join(stateDir, `${id}.json`), "utf8")).captureFormatVersion, CAPTURE_FORMAT_VERSION);
  }
});

test("partial sends retry only the unsent suffix with identical source payloads", async (t) => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-codex-source-retry-"));
  const previous = process.env.OPENVIKING_CODEX_STATE_DIR;
  process.env.OPENVIKING_CODEX_STATE_DIR = stateDir;
  t.after(async () => {
    if (previous === undefined) delete process.env.OPENVIKING_CODEX_STATE_DIR;
    else process.env.OPENVIKING_CODEX_STATE_DIR = previous;
    await rm(stateDir, { recursive: true, force: true });
  });
  const transcriptPath = join(stateDir, "retry.jsonl");
  const records = Array.from({ length: 101 }, (_, i) => message(`Source message ${i}`, { ...source, source_message_ids: [`source-${i}`] }));
  await writeFile(transcriptPath, records.map(JSON.stringify).join("\n"));
  const state = await loadState("retry");
  const attempts = [];
  const fetchJSONRes = async (_path, init) => {
    const body = JSON.parse(init.body);
    attempts.push(body.messages);
    return { ok: attempts.length !== 2, status: attempts.length === 2 ? 503 : 200 };
  };
  assert.equal((await catchUpTurns({ state, transcriptPath, cfg, fetchJSONRes })).added, 100);
  assert.equal((await loadState("retry")).capturedTurnCount, 100);
  const resumed = await loadState("retry");
  assert.equal((await catchUpTurns({ state: resumed, transcriptPath, cfg, fetchJSONRes })).added, 1);
  assert.deepEqual(attempts[1], attempts[2]);
  assert.deepEqual(attempts[2], [{ role: "user", parts: [{ type: "text", text: "Source message 100" }], ...source, source_message_ids: ["source-100"] }]);
  assert.equal(resumed.capturedTurnCount, 101);
});
