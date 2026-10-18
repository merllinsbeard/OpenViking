export class OriginContractMismatch extends Error {}
export function originStatements(value: unknown): { version: 1; contributors: Array<Record<string, unknown>> };
export function observedTime(
  literal: string | number,
  meaning: "source_event" | "provider_record" | "host_stamp" | "unknown",
  basis: { producer: string; field: string },
): { literal: string | number; encoding: "rfc3339" | "unix_seconds" | "opaque"; meaning: typeof meaning; basis: typeof basis };
export function validateOriginReadback(messages: Array<Record<string, any>>, result: unknown): void;
