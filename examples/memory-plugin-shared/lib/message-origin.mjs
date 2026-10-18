export class OriginContractMismatch extends Error {
  constructor(message) {
    super(message);
    this.name = "OriginContractMismatch";
  }
}

function requireValue(condition, message) {
  if (!condition) throw new OriginContractMismatch(message);
}

function object(value, required, optional = []) {
  requireValue(value && typeof value === "object" && !Array.isArray(value), "Expected an origin object");
  requireValue(required.every((key) => Object.hasOwn(value, key))
    && Object.keys(value).every((key) => required.includes(key) || optional.includes(key)), "Incomplete or unsupported origin fields");
  return value;
}

function text(value) {
  requireValue(typeof value === "string" && value.length > 0, "Expected a nonempty origin string");
  return value;
}

function optionalText(value) {
  requireValue(value == null || typeof value === "string", "Invalid optional origin string");
  return value ?? null;
}

function rfc3339(value) {
  text(value);
  requireValue(/^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:(?:[0-5]\d|60)(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$/.test(value), "Invalid RFC3339 origin time");
  const check = value.slice(0, 17) + (value.slice(17, 19) === "60" ? "59" : value.slice(17, 19)) + value.slice(19);
  const [year, month, day] = value.slice(0, 10).split("-").map(Number);
  const days = new Date(Date.UTC(year, month, 0)).getUTCDate();
  requireValue(month >= 1 && month <= 12 && day >= 1 && day <= days && Number.isFinite(Date.parse(check)), "Invalid RFC3339 origin date");
  return value;
}

export function observedTime(literal, meaning, fieldBasis) {
  let encoding = typeof literal === "number" ? "unix_seconds" : "opaque";
  if (typeof literal === "string") {
    try {
      rfc3339(literal);
      encoding = "rfc3339";
    } catch (error) {
      if (!(error instanceof OriginContractMismatch)) throw error;
    }
  }
  return { literal, encoding, meaning, basis: fieldBasis };
}

function basis(value) {
  object(value, ["producer", "field"]);
  return { producer: text(value.producer), field: text(value.field) };
}

function actor(value) {
  if (value == null) return null;
  object(value, ["actor", "basis"]);
  object(value.actor, ["identifier"], ["scope"]);
  return { actor: { identifier: text(value.actor.identifier), scope: optionalText(value.actor.scope) }, basis: basis(value.basis) };
}

export function originStatements(value) {
  object(value, ["version", "contributors"]);
  requireValue(value.version === 1 && Array.isArray(value.contributors), "Unsupported origin schema");
  return {
    version: 1,
    contributors: value.contributors.map((item) => {
      object(item, ["locator"], ["refs", "times", "transport_sender", "primary_author", "content_class", "class_basis"]);
      object(item.locator, ["provider"], ["session_id", "source_scope"]);
      requireValue(Array.isArray(item.refs ?? []) && Array.isArray(item.times ?? []), "Origin refs/times must be arrays");
      const refs = (item.refs ?? []).map((ref) => {
        object(ref, ["kind", "value"]);
        requireValue(["provider_record", "host_row", "source_message", "supplied_ref"].includes(ref.kind), "Unsupported origin reference");
        return { kind: ref.kind, value: text(ref.value) };
      });
      const times = (item.times ?? []).map((time) => {
        object(time, ["literal", "encoding", "meaning", "basis"]);
        const numeric = typeof time.literal === "number" && Number.isFinite(time.literal);
        requireValue(["source_event", "provider_record", "host_stamp", "unknown"].includes(time.meaning), "Unsupported origin time meaning");
        if (time.encoding === "rfc3339") rfc3339(time.literal);
        else requireValue(time.encoding === "unix_seconds" ? numeric : time.encoding === "opaque" && (numeric || typeof time.literal === "string"), "Invalid origin time literal");
        return { ...time, basis: basis(time.basis) };
      });
      const contentClass = item.content_class ?? "unknown";
      requireValue(["unknown", "direct_input", "forwarded", "runtime_self_metadata", "bb_relay"].includes(contentClass), "Unsupported origin content class");
      requireValue(contentClass === "unknown" || item.class_basis != null, "Known origin class needs a native basis");
      return {
        locator: { provider: text(item.locator.provider), session_id: optionalText(item.locator.session_id), source_scope: optionalText(item.locator.source_scope) },
        refs, times, transport_sender: actor(item.transport_sender), primary_author: actor(item.primary_author),
        content_class: contentClass, class_basis: item.class_basis == null ? null : basis(item.class_basis),
      };
    }),
  };
}

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  return JSON.stringify(value);
}

/** Validate the persistent members before a sender advances its contiguous prefix. */
export function validateOriginReadback(messages, result) {
  if (!messages.some((message) => message.origin != null)) return;
  requireValue(result?.origin_readback === "jsonl" && Array.isArray(result.accepted_origins), "Server omitted persistent origin readback");
  const receipts = result.accepted_origins;
  const ids = new Set();
  const groups = new Set();
  const stamps = new Set();
  for (const [inputIndex, message] of messages.entries()) {
    const parts = message.parts ?? [];
    const count = message.role === "user" && parts.length > 1 && parts.every((part) => part.type === "tool") ? parts.length : 1;
    const members = receipts.filter((item) => Number.isInteger(item?.input_index) && item.input_index === inputIndex);
    requireValue(members.length === count, "Incomplete origin input/fanout coverage");
    const expected = canonical(originStatements(message.origin ?? { version: 1, contributors: [] }));
    const indices = new Set();
    const memberGroups = new Set();
    for (const member of members) {
      object(member, ["input_index", "message_id", "origin", "legacy_created_at", "turn_id", "message_kind", "source_message_ids"]);
      text(member.message_id);
      requireValue(!ids.has(member.message_id), "Duplicate origin message ID");
      ids.add(member.message_id);
      object(member.origin, ["version", "contributors", "receiver"]);
      const { receiver, ...statements } = member.origin;
      requireValue(canonical(originStatements(statements)) === expected, "Stored contributor literals changed");
      object(receiver, ["session_ingest_started_at", "applicability"]);
      stamps.add(rfc3339(receiver.session_ingest_started_at));
      const scope = object(receiver.applicability, ["kind", "group_id"], ["member_index", "member_count"]);
      memberGroups.add(text(scope.group_id));
      if (count === 1) requireValue(scope.kind === "input_message" && !Object.hasOwn(scope, "member_index") && !Object.hasOwn(scope, "member_count"), "Invalid whole-input scope");
      else {
        requireValue(scope.kind === "input_group" && scope.member_count === count && Number.isInteger(scope.member_index) && scope.member_index >= 0 && scope.member_index < count && !indices.has(scope.member_index), "Invalid fanout scope");
        indices.add(scope.member_index);
      }
      const kind = message.message_kind || (count > 1 ? "tool_transport" : null);
      requireValue(member.legacy_created_at === (message.created_at || receiver.session_ingest_started_at)
        && member.turn_id === (message.turn_id ?? null) && member.message_kind === kind
        && canonical(member.source_message_ids) === canonical(message.source_message_ids ?? null), "Stored source fields changed");
    }
    requireValue(memberGroups.size === 1 && !groups.has([...memberGroups][0]), "Origin group identity reused");
    groups.add([...memberGroups][0]);
  }
  requireValue(ids.size === receipts.length && stamps.size === 1, "Unexpected origin members or clocks");
}
