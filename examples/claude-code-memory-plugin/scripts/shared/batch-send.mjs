// GENERATED FROM examples/memory-plugin-shared/lib. DO NOT EDIT.
import { enqueue } from "./pending-queue.mjs";
import { isRetryableFailure } from "./retryable.mjs";
import { OriginContractMismatch, originStatements, validateOriginReadback } from "./message-origin.mjs";

export const BATCH_LIMIT = 100;

export function isRetryableSendFailure(res) {
  return isRetryableFailure(res);
}

function makeResult() {
  return {
    sent: 0,
    queued: 0,
    enqueueFailed: 0,
    failed: 0,
    retryable: false,
    usedBatch: true,
    lastError: null,
  };
}

function originFailure(result, remaining, error) {
  result.failed += remaining;
  result.retryable = false;
  result.lastError = error instanceof OriginContractMismatch
    ? error : new OriginContractMismatch("Origin append outcome is uncertain");
  return result;
}

async function enqueueRemainder(sessionId, payloads, startIndex, result, retryable, lastError) {
  result.retryable = retryable;
  result.lastError = lastError ?? null;

  if (!retryable) {
    result.failed += Math.max(0, payloads.length - startIndex);
    return result;
  }

  const baseCreatedAt = Date.now();
  for (let i = startIndex; i < payloads.length; i++) {
    const queued = await enqueue("addMessage", sessionId, payloads[i], {
      createdAt: baseCreatedAt + (i - startIndex),
    });
    if (!queued.ok) {
      // Stop at the first enqueue failure so queued entries stay a contiguous
      // prefix: consumers mark the first sent+queued payloads as captured, so
      // skipping a payload here and queueing a later one would silently drop it.
      result.enqueueFailed += payloads.length - i;
      return result;
    }
    result.queued++;
  }
  return result;
}

async function sendSerial(fetchJSON, sessionId, payloads, startIndex, opts, result) {
  result.usedBatch = false;
  const encodedSid = encodeURIComponent(sessionId);
  for (let i = startIndex; i < payloads.length; i++) {
    let res;
    try {
      res = await fetchJSON(`/api/v1/sessions/${encodedSid}/messages`, {
        method: "POST",
        body: JSON.stringify(payloads[i]),
      });
      if (res?.ok) validateOriginReadback([payloads[i]], res.result);
    } catch (error) {
      if (payloads[i].origin == null) throw error;
      return originFailure(result, payloads.length - i, error);
    }
    if (res?.ok) {
      result.sent++;
      await opts.onSent?.(1);
      continue;
    }

    if (payloads[i].origin != null) return originFailure(result, payloads.length - i, res?.error);
    const retryable = isRetryableSendFailure(res);
    if (opts.enqueueOnRetryable) {
      return enqueueRemainder(sessionId, payloads, i, result, retryable, res?.error ?? res);
    }
    result.retryable = retryable;
    result.lastError = res?.error ?? res ?? null;
    result.failed += payloads.length - i;
    return result;
  }
  return result;
}

/**
 * Send add-message payloads in server-sized batches with serial fallback.
 *
 * @param {Function} fetchJSON - (path, init) => { ok, status, result?, error? }
 * @param {string} sessionId - OpenViking session id
 * @param {Array<object>} payloads - sanitized add-message request bodies
 * @param {object} opts
 * @param {boolean} opts.enqueueOnRetryable - enqueue unsent payloads after a retryable failure
 * @param {Function} opts.onSent - called with the number of messages durably sent after each success
 * @returns {Promise<{sent:number,queued:number,enqueueFailed:number,failed:number,retryable:boolean,usedBatch:boolean,lastError:any}>}
 */
export async function sendSessionMessages(fetchJSON, sessionId, payloads, opts = {}) {
  const result = makeResult();
  const messages = Array.isArray(payloads) ? payloads : [];
  if (messages.length === 0) return result;
  try {
    for (const message of messages) {
      if (message.origin != null) originStatements(message.origin);
    }
  } catch (error) {
    return originFailure(result, messages.length, error);
  }

  const encodedSid = encodeURIComponent(sessionId);
  for (let start = 0; start < messages.length; start += BATCH_LIMIT) {
    const chunk = messages.slice(start, start + BATCH_LIMIT);
    let res;
    const requiresOrigin = chunk.some((message) => message.origin != null);
    try {
      res = await fetchJSON(`/api/v1/sessions/${encodedSid}/messages/batch`, {
        method: "POST",
        body: JSON.stringify({ messages: chunk }),
      });
      if (res?.ok) validateOriginReadback(chunk, res.result);
    } catch (error) {
      if (!requiresOrigin) throw error;
      return originFailure(result, messages.length - start, error);
    }

    if (res?.ok) {
      result.sent += chunk.length;
      await opts.onSent?.(chunk.length);
      continue;
    }

    const status = Number(res?.status || 0);
    if (status === 404 || status === 405) {
      return sendSerial(fetchJSON, sessionId, messages, start, opts, result);
    }

    if (requiresOrigin) return originFailure(result, messages.length - start, res?.error);
    const retryable = isRetryableSendFailure(res);
    if (opts.enqueueOnRetryable) {
      return enqueueRemainder(sessionId, messages, start, result, retryable, res?.error ?? res);
    }
    result.retryable = retryable;
    result.lastError = res?.error ?? res ?? null;
    result.failed += messages.length - start;
    return result;
  }

  return result;
}
