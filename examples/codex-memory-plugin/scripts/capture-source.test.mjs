import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { mockOriginReadback } from "../../memory-plugin-shared/testing/support.mjs";

import { extractCaptureTranscript, extractCaptureTurns } from "./capture-utils.mjs";
import { catchUpTurns } from "./ov-session.mjs";
import { CAPTURE_FORMAT_VERSION, loadState, saveState } from "./session-state.mjs";

const cfg = { captureAssistantTurns: true };

function legacyTurn({ origin: _origin, ...turn }) { return turn; }
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
  assert.deepEqual(turns.map(legacyTurn), [{
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
  assert.deepEqual(turns.slice(0, 2).map(legacyTurn), [
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
    payload: { operation: "append", turn_id: "history-turn", source_message_ids: ["history-wrapper"], items: [
      { type: "message", role: "user", content: [{ type: "input_text", text: "History question" }],
        id: "history-message", created_at: source.created_at, source_message_ids: ["history-source"], message_kind: "checkpoint" },
    ] },
  }], cfg);
  assert.deepEqual(turns.map(legacyTurn), [{
    role: "user", text: "History question", parts: [{ type: "text", text: "History question" }],
    created_at: source.created_at, turn_id: "history-turn", message_kind: "checkpoint",
    source_message_ids: ["history-wrapper", "history-source", "history-message"],
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

test("conflicting scalars stay absent after response replay and completed-tool dedup", () => {
  const call = (timestamp, turn_id, message_kind) => ({ timestamp, type: "response_item", payload: {
    type: "function_call", call_id: "conflicting-call", name: "exec_command", arguments: { cmd: "pwd" }, turn_id, message_kind,
  } });
  const turns = extractCaptureTurns([
    call(source.created_at, "first-turn", "assistant_step"),
    call("2026-10-18T00:00:00Z", "second-turn", "tool_transport"),
    { timestamp: source.created_at, type: "event_msg", payload: {
      type: "exec_command_end", call_id: "conflicting-call", command: "pwd", aggregated_output: "project root",
      turn_id: "first-turn", message_kind: "assistant_step", source_message_ids: ["completed-source"],
    } },
  ], cfg);
  assert.equal(turns.length, 2);
  for (const turn of turns) {
    assert.equal(turn.created_at, undefined);
    assert.equal(turn.turn_id, undefined);
    assert.equal(turn.message_kind, undefined);
    assert.deepEqual(turn.source_message_ids, ["completed-source"]);
  }
  assert.equal(turns[1].parts[0].tool_output, "project root");
});

function completedGroups() {
  return [
    { timestamp: "2026-10-17T23:48:12.123456+05:30", type: "event_msg", payload: {
      type: "item_completed", item: {
        type: "CommandExecution", id: "same-id", command: "pwd", aggregated_output: "command result", status: "completed",
        source_message_ids: ["command-ref"], turn_id: "command-turn", message_kind: "assistant_step",
      },
    } },
    { timestamp: "2026-10-18T00:00:00Z", type: "event_msg", payload: {
      type: "item_completed", item: {
        type: "DynamicToolCall", id: "same-id", tool: "exec_command", arguments: { cmd: "pwd" },
        content_items: [{ type: "inputText", text: "dynamic result" }], status: "completed",
        source_message_ids: ["dynamic-ref"], turn_id: "dynamic-turn", message_kind: "tool_transport",
      },
    } },
  ];
}

const separateGroupSources = [
  { created_at: "2026-10-17T23:48:12.123456+05:30", turn_id: "command-turn", message_kind: "assistant_step", source_message_ids: ["command-ref", "same-id"] },
  { created_at: "2026-10-17T23:48:12.123456+05:30", turn_id: "command-turn", message_kind: "assistant_step", source_message_ids: ["command-ref", "same-id"] },
  { created_at: "2026-10-18T00:00:00Z", turn_id: "dynamic-turn", message_kind: "tool_transport", source_message_ids: ["dynamic-ref", "same-id"] },
  { created_at: "2026-10-18T00:00:00Z", turn_id: "dynamic-turn", message_kind: "tool_transport", source_message_ids: ["dynamic-ref", "same-id"] },
];

function sourceFields(turn) {
  return Object.fromEntries(["created_at", "turn_id", "message_kind", "source_message_ids"]
    .filter((field) => turn[field] !== undefined).map((field) => [field, turn[field]]));
}

test("keeps independent sources for completed groups with the same id and name across kinds", () => {
  const { turns, excludedLegacyTurnIndices } = extractCaptureTranscript(completedGroups(), cfg);
  assert.deepEqual(turns.map(sourceFields), separateGroupSources);
  assert.deepEqual(turns.map((turn) => turn.parts), [
    [{ type: "tool", tool_id: "same-id", tool_name: "exec_command", tool_status: "running", tool_input: { command: "pwd" } }],
    [{ type: "tool", tool_id: "same-id", tool_name: "exec_command", tool_status: "completed", tool_output: "command result" }],
    [{ type: "tool", tool_id: "same-id", tool_name: "exec_command", tool_status: "running", tool_input: { cmd: "pwd" } }],
    [{ type: "tool", tool_id: "same-id", tool_name: "exec_command", tool_status: "completed", tool_output: '[{"type":"inputText","text":"dynamic result"}]' }],
  ]);
  assert.deepEqual(excludedLegacyTurnIndices, []);
});

for (const name of [undefined, "exec_command"]) {
  test(`does not attach a ${name ? "named" : "nameless"} response to multiple completed kinds`, () => {
    const turns = extractCaptureTurns([
      { timestamp: "2026-10-18T00:00:01Z", type: "response_item", payload: {
        type: "function_call_output", call_id: "same-id", id: "ambiguous-response", name, output: "response result",
        turn_id: "response-turn", message_kind: "checkpoint", source_message_ids: ["response-ref"],
      } },
      ...completedGroups(),
    ], cfg);
    assert.deepEqual(turns.map(sourceFields), separateGroupSources);
  });
}

test("attaches response sources only to the unique full group among names and keeps completed duplicates", () => {
  const groups = completedGroups();
  groups[1].payload.item.tool = "read_file";
  const turns = extractCaptureTurns([
    { timestamp: "2026-10-17T23:48:12.123456+05:30", type: "response_item", payload: {
      type: "function_call_output", call_id: "same-id", id: "named-response", name: "exec_command", output: "old result",
      turn_id: "command-turn", message_kind: "assistant_step", source_message_ids: ["response-ref"],
    } },
    ...groups,
    { type: "event_msg", payload: {
      type: "exec_command_end", call_id: "same-id", command: "old command", aggregated_output: "old result",
      source_message_ids: ["command-duplicate"],
    } },
  ], cfg);
  assert.deepEqual(turns.map(sourceFields), [
    { created_at: "2026-10-17T23:48:12.123456+05:30", turn_id: "command-turn", message_kind: "assistant_step", source_message_ids: ["command-ref", "same-id", "command-duplicate", "response-ref", "named-response"] },
    { created_at: "2026-10-17T23:48:12.123456+05:30", turn_id: "command-turn", message_kind: "assistant_step", source_message_ids: ["command-ref", "same-id", "command-duplicate", "response-ref", "named-response"] },
    { created_at: "2026-10-18T00:00:00Z", turn_id: "dynamic-turn", message_kind: "tool_transport", source_message_ids: ["dynamic-ref", "same-id"] },
    { created_at: "2026-10-18T00:00:00Z", turn_id: "dynamic-turn", message_kind: "tool_transport", source_message_ids: ["dynamic-ref", "same-id"] },
  ]);
  assert.deepEqual(turns.map((turn) => turn.parts[0].tool_name), ["exec_command", "read_file", "read_file", "read_file"]);
  assert.equal(turns[1].parts[0].tool_output, "command result");
});

test("catchUpTurns keeps separate group payloads and retries only the unsent suffix with MOCK HTTP", async (t) => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-codex-source-groups-"));
  const previous = process.env.OPENVIKING_CODEX_STATE_DIR;
  process.env.OPENVIKING_CODEX_STATE_DIR = stateDir;
  t.after(async () => {
    if (previous === undefined) delete process.env.OPENVIKING_CODEX_STATE_DIR;
    else process.env.OPENVIKING_CODEX_STATE_DIR = previous;
    await rm(stateDir, { recursive: true, force: true });
  });
  const expected = [
    { role: "assistant", parts: [{ type: "tool", tool_id: "same-id", tool_name: "exec_command", tool_status: "running", tool_input: { command: "pwd" } }], ...separateGroupSources[0] },
    { role: "user", parts: [{ type: "tool", tool_id: "same-id", tool_name: "exec_command", tool_status: "completed", tool_output: "command result" }], ...separateGroupSources[1] },
    { role: "assistant", parts: [{ type: "tool", tool_id: "same-id", tool_name: "exec_command", tool_status: "running", tool_input: { cmd: "pwd" } }], ...separateGroupSources[2] },
    { role: "user", parts: [{ type: "tool", tool_id: "same-id", tool_name: "exec_command", tool_status: "completed", tool_output: '[{"type":"inputText","text":"dynamic result"}]' }], ...separateGroupSources[3] },
  ];
  for (const serial of [false, true]) {
    const id = `groups-${serial}`;
    const transcriptPath = join(stateDir, `${id}.jsonl`);
    await writeFile(transcriptPath, [
      { type: "response_item", timestamp: "2026-10-18T00:00:01Z", payload: {
        type: "function_call_output", call_id: "same-id", id: "unattributed", name: "exec_command", output: "response result",
      } },
      ...completedGroups(),
    ].map(JSON.stringify).join("\n"));
    const state = await loadState(id);
    assert.equal(state.capturedTurnCount, 0);
    const attempts = [];
    const sent = [];
    let failed = false;
    const fetchJSONRes = async (path, init) => {
      const body = JSON.parse(init.body);
      attempts.push({ path, body });
      if (serial && path.endsWith("/batch")) return { ok: false, status: 404 };
      if (serial && sent.length === 2 && !failed) {
        failed = true;
        return { ok: false, status: 503 };
      }
      sent.push(...(body.messages ?? [body]));
      return { ok: true, status: 200, result: mockOriginReadback(body.messages ?? [body]) };
    };
    assert.equal((await catchUpTurns({ state, transcriptPath, cfg, fetchJSONRes })).added, serial ? 2 : 4);
    assert.equal((await loadState(id)).capturedTurnCount, serial ? 2 : 4);
    if (serial) {
      const resumed = await loadState(id);
      assert.equal((await catchUpTurns({ state: resumed, transcriptPath, cfg, fetchJSONRes })).added, 2);
      assert.deepEqual(attempts[3].body, attempts[5].body);
    }
    assert.deepEqual(sent.map(legacyTurn), expected);
    const count = attempts.length;
    assert.equal((await loadState(id)).capturedTurnCount, 4);
    assert.equal((await catchUpTurns({ state: await loadState(id), transcriptPath, cfg, fetchJSONRes })).added, 0);
    assert.equal(attempts.length, count);
    await writeFile(transcriptPath, `${await readFile(transcriptPath, "utf8")}\n${JSON.stringify(message("Appended tail", { id: "tail-id" }))}`);
    assert.equal((await catchUpTurns({ state: await loadState(id), transcriptPath, cfg, fetchJSONRes })).added, 1);
    assert.equal((await loadState(id)).capturedTurnCount, 5);
    assert.deepEqual(legacyTurn(sent[4]), { role: "user", parts: [{ type: "text", text: "Appended tail" }], created_at: "2026-10-17T23:48:12.123456+05:30", source_message_ids: ["tail-id"] });
  }
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
    const records = [message("Captured prefix"), message("New source message", source),
      { ...message("Unknown payload metadata"), timestamp: undefined }];
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
      return { ok: true, status: 200, result: mockOriginReadback(body.messages ?? [body]) };
    };
    const first = await catchUpTurns({ state, transcriptPath, cfg, activePeerId: "project", fetchJSONRes });
    assert.equal(first.added, 2);
    assert.deepEqual(sent.map(legacyTurn), [
      { role: "user", parts: [{ type: "text", text: "New source message" }], peer_id: "project", ...source },
      { role: "user", parts: [{ type: "text", text: "Unknown payload metadata" }], peer_id: "project" },
    ]);
    assert.equal((await loadState(id)).capturedTurnCount, 3);
    assert.equal((await catchUpTurns({ state, transcriptPath, cfg, fetchJSONRes })).added, 0);
    assert.equal(sent.length, 2);
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
    return { ok: attempts.length !== 2, status: attempts.length === 2 ? 503 : 200, result: mockOriginReadback(body.messages) };
  };
  assert.equal((await catchUpTurns({ state, transcriptPath, cfg, fetchJSONRes })).added, 100);
  assert.equal((await loadState("retry")).capturedTurnCount, 100);
  const resumed = await loadState("retry");
  assert.equal((await catchUpTurns({ state: resumed, transcriptPath, cfg, fetchJSONRes })).added, 1);
  assert.deepEqual(attempts[1], attempts[2]);
  assert.deepEqual(attempts[2].map(legacyTurn), [{ role: "user", parts: [{ type: "text", text: "Source message 100" }], ...source, source_message_ids: ["source-100"] }]);
  assert.equal(resumed.capturedTurnCount, 101);
});

test("native capture keeps record time separate from an explicit unknown source time", () => {
  const turn = extractCaptureTurns([message("Forwarded words", {
    id: "native-A", created_at: "2020-01-01T01:02:03.123456+07:00",
  })], cfg)[0];
  assert.equal(turn.created_at, "2020-01-01T01:02:03.123456+07:00");
  assert.deepEqual(turn.origin.contributors[0].times, [
    { literal: "2026-10-17T23:48:12.123456+05:30", encoding: "rfc3339", meaning: "provider_record", basis: { producer: "codex", field: "entry.timestamp" } },
    { literal: "2020-01-01T01:02:03.123456+07:00", encoding: "rfc3339", meaning: "unknown", basis: { producer: "codex", field: "payload.created_at" } },
  ]);
  assert.equal(turn.origin.contributors[0].primary_author, null);
  assert.equal(turn.origin.contributors[0].content_class, "unknown");
});

test("history wrapper observations and missing child observations remain separate", () => {
  const turn = extractCaptureTurns([{
    type: "history_mutation", timestamp: "2026-10-17T00:00:00Z",
    payload: { operation: "append", source_message_ids: ["wrapper"], items: [
      { type: "message", role: "user", id: "child", content: [{ type: "input_text", text: "Unmarked child" }] },
    ] },
  }], cfg)[0];
  const contributors = turn.origin.contributors;
  assert.equal(contributors.length, 2);
  assert.deepEqual(contributors[0].refs, [{ kind: "supplied_ref", value: "wrapper" }]);
  assert.equal(contributors[0].times[0].meaning, "provider_record");
  assert.deepEqual(contributors[1].refs, [{ kind: "provider_record", value: "child" }]);
  assert.deepEqual(contributors[1].times, []);
});

test("undeclared legacy time keeps its literal when it is not a valid RFC3339 date", () => {
  const turn = extractCaptureTurns([message("Opaque source time", { created_at: "2026-13-32T00:00:00Z" })], cfg)[0];
  assert.equal(turn.created_at, "2026-13-32T00:00:00Z");
  assert.deepEqual(turn.origin.contributors[0].times[1], {
    literal: "2026-13-32T00:00:00Z", encoding: "opaque", meaning: "unknown",
    basis: { producer: "codex", field: "payload.created_at" },
  });
});

for (const failure of ["omitted", "partial", "altered", "strict", "lost"]) {
  test(`actual catchUpTurns holds its persisted cursor on ${failure} origin ACK`, async (t) => {
    const stateDir = await mkdtemp(join(tmpdir(), "ov-codex-origin-"));
    const previous = process.env.OPENVIKING_CODEX_STATE_DIR;
    process.env.OPENVIKING_CODEX_STATE_DIR = stateDir;
    t.after(async () => {
      if (previous === undefined) delete process.env.OPENVIKING_CODEX_STATE_DIR;
      else process.env.OPENVIKING_CODEX_STATE_DIR = previous;
      await rm(stateDir, { recursive: true, force: true });
    });
    const transcriptPath = join(stateDir, "source.jsonl");
    await writeFile(transcriptPath, [message("Already captured"), message("New literal", source)].map(JSON.stringify).join("\n"));
    const state = await loadState("native-session");
    state.capturedTurnCount = 1;
    state.captureFormatVersion = 2;
    await saveState(state);
    let calls = 0;
    const result = await catchUpTurns({ state, transcriptPath, cfg, fetchJSONRes: async (_path, init) => {
      calls++;
      const body = JSON.parse(init.body);
      assert.equal(body.messages[0].origin.contributors[0].locator.session_id, "native-session");
      if (failure === "lost") throw new Error("MOCK ACK lost after append");
      if (failure === "strict") return { ok: false, status: 422 };
      const receipt = mockOriginReadback(body.messages);
      if (failure === "omitted") delete receipt.accepted_origins;
      if (failure === "partial") delete receipt.accepted_origins[0].origin.receiver;
      if (failure === "altered") receipt.accepted_origins[0].origin.contributors[0].times[0].literal = "1900-01-01T00:00:00Z";
      return { ok: true, status: 200, result: receipt };
    } });
    assert.equal(calls, 1);
    assert.equal(result.added, 0);
    assert.equal((await loadState("native-session")).capturedTurnCount, 1);
  });
}
