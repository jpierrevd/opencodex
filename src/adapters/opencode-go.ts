import { customToolWireName } from "../responses/custom-tool-compat";

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

/** Match the Go destination, including user-renamed provider entries. */
export function isOpenCodeGo(baseUrl: string): boolean {
  try {
    const url = new URL(baseUrl);
    return url.origin === "https://opencode.ai" && url.pathname.replace(/\/+$/, "") === "/zen/go/v1";
  } catch { return false; }
}

/** Plaintext part types Console Go accepts inside a converted message. */
const GO_PLAINTEXT_PART_TYPES = ["input_text", "input_image", "input_file"];

/** Wire-safe content part check for Console Go message conversion. */
function isGoPlaintextPart(part: unknown): boolean {
  return !!part && typeof part === "object" && !Array.isArray(part)
    && GO_PLAINTEXT_PART_TYPES.includes((part as { type?: unknown }).type as string);
}

/** Dedupe identity for promoted declarations: type plus wire name (namespace-aware). */
function toolIdentityKey(tool: unknown): string | undefined {
  if (!tool || typeof tool !== "object" || Array.isArray(tool)) return undefined;
  const rec = tool as { type?: unknown; name?: unknown; namespace?: unknown };
  if (typeof rec.type !== "string" || typeof rec.name !== "string") return undefined;
  // Compare by wire identity so a flat declaration and the same tool inside the
  // builtin `functions` namespace group dedupe instead of doubling upstream,
  // where duplicate function names are rejected.
  return `${rec.type}\n${customToolWireName(typeof rec.namespace === "string" ? rec.namespace : undefined, rec.name)}`;
}

/**
 * Promote Codex Desktop's responses-lite `additional_tools` input items to top-level
 * `tools` and drop the items. The parser already collects these declarations into the
 * tool surface, but the outbound body keeps the item verbatim and Console Go's validator
 * rejects the unknown item type (`input[N] did not match any supported type`). Promoting
 * preserves every declaration (deduplicated by wire identity, descending into namespace
 * groups so a flat declaration and the same tool inside a group do not double upstream)
 * in the standard shape the downstream namespace/custom lowering passes already handle.
 */
export function normalizeOpenCodeGoAdditionalTools(body: unknown): unknown {
  if (!body || typeof body !== "object" || Array.isArray(body)) return body;
  const record = body as Record<string, unknown>;
  if (!Array.isArray(record.input)) return body;
  const existing = Array.isArray(record.tools) ? (record.tools as unknown[]) : [];
  const seen = new Set<string>();
  // Claim a child declaration; returns false for duplicates and unidentifiable
  // entries. Claiming inside the filter (rather than after it) keeps two equal
  // children of the same container from both surviving.
  const claim = (child: unknown): boolean => {
    const key = toolIdentityKey(child);
    if (key === undefined || seen.has(key)) return false;
    seen.add(key);
    return true;
  };
  const markSeen = (tool: unknown): void => {
    if (!isRecord(tool)) return;
    if (tool.type === "namespace" && typeof tool.name === "string" && Array.isArray(tool.tools)) {
      for (const child of tool.tools) claim(child);
      return;
    }
    claim(tool);
  };
  // Output buckets: top-level declarations, with at most one container per
  // namespace name. Existing containers are copied before merging so the
  // caller's declarations are never mutated.
  const outTools: unknown[] = [];
  const groupSlot = new Map<string, number>();
  for (const tool of existing) {
    if (isRecord(tool) && tool.type === "namespace" && typeof tool.name === "string" && Array.isArray(tool.tools)) {
      for (const child of tool.tools as unknown[]) claim(child);
      groupSlot.set(tool.name, outTools.length);
    } else {
      claim(tool);
    }
    outTools.push(tool);
  }
  const mergeGroup = (name: string, first: Record<string, unknown>, kept: unknown[]): void => {
    const slot = groupSlot.get(name);
    if (slot === undefined) {
      const group = { ...first, tools: [] as unknown[] };
      groupSlot.set(name, outTools.length);
      outTools.push(group);
      (group.tools as unknown[]).push(...kept);
      return;
    }
    const current = outTools[slot] as Record<string, unknown>;
    outTools[slot] = { ...current, tools: [...(current.tools as unknown[]), ...kept] };
  };
  const promote = (tool: unknown): unknown | undefined => {
    if (!isRecord(tool)) return undefined;
    // Merge matching namespace containers so distinct children from multiple
    // additional_tools items (and pre-existing top-level containers) land in a
    // single container instead of several same-named groups downstream.
    if (tool.type === "namespace" && typeof tool.name === "string" && Array.isArray(tool.tools)) {
      const kept = (tool.tools as unknown[]).filter(child => claim(child));
      if (kept.length === 0) return undefined;
      mergeGroup(tool.name, tool, kept);
      return undefined;
    }
    return claim(tool) ? tool : undefined;
  };
  let changed = false;
  const input: unknown[] = [];
  for (const item of record.input as unknown[]) {
    if (!isRecord(item)
      || item.type !== "additional_tools"
      || !Array.isArray(item.tools)) {
      input.push(item);
      continue;
    }
    changed = true;
    // Entries without a type/name identity cannot be matched by any downstream
    // pass (namespace/custom lowering and tool_choice filtering all key on them);
    // promoting them would only add a guaranteed-400 entry on a closed validator.
    for (const tool of item.tools as unknown[]) {
      const next = promote(tool);
      if (next !== undefined) outTools.push(next);
    }
  }
  if (!changed) return body;
  return { ...record, input, tools: outTools };
}

/** Public Responses rejects Codex's private agent_message variant, even with plaintext content. */
export function normalizeOpenCodeGoAgentMessages(body: unknown): unknown {
  if (!body || typeof body !== "object" || Array.isArray(body)) return body;
  const record = body as Record<string, unknown>;
  if (!Array.isArray(record.input)) return body;
  let changed = false;
  const input = record.input.map((item: unknown) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return item;
    const message = item as Record<string, unknown>;
    if (message.type !== "agent_message" || !Array.isArray(message.content) || message.content.length === 0) return item;
    // Genuine ciphertext and unknown part types must retain their existing fail-closed path
    // when no plaintext survives: an empty message would be rejected too. But a MIXED item
    // (plaintext task envelope beside inter-agent ciphertext) must not fail the whole
    // request: Console Go can never decode ciphertext minted for another Codex agent, and
    // its validator rejects the unknown agent_message type outright. Convert carrying only
    // the plaintext parts so the task envelope still reaches the model.
    const plaintext = (message.content as unknown[]).filter(isGoPlaintextPart);
    if (plaintext.length === 0) return item;
    const content = plaintext.length === (message.content as unknown[]).length
      ? (message.content as unknown[])
      : plaintext;
    const identities = Object.fromEntries(["author", "recipient"]
      .filter(key => typeof message[key] === "string")
      .map(key => [key, message[key]]));
    changed = true;
    return {
      type: "message", role: "user",
      content: [
        ...(Object.keys(identities).length ? [{ type: "input_text", text: `Agent message ${JSON.stringify(identities)}` }] : []),
        ...content,
      ],
    };
  });
  return changed ? { ...record, input } : body;
}
