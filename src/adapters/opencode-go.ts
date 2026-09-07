/** Match the Go destination, including user-renamed provider entries. */
export function isOpenCodeGo(baseUrl: string): boolean {
  try {
    const url = new URL(baseUrl);
    return url.origin === "https://opencode.ai" && url.pathname.replace(/\/+$/, "") === "/zen/go/v1";
  } catch { return false; }
}

/** Plaintext part types Console Go accepts inside a converted message. */
const GO_PLAINTEXT_PART_TYPES = ["input_text", "input_image", "input_file"];

function isGoPlaintextPart(part: unknown): boolean {
  return !!part && typeof part === "object" && !Array.isArray(part)
    && GO_PLAINTEXT_PART_TYPES.includes((part as { type?: unknown }).type as string);
}

function toolIdentityKey(tool: unknown): string | undefined {
  if (!tool || typeof tool !== "object" || Array.isArray(tool)) return undefined;
  const rec = tool as { type?: unknown; name?: unknown; namespace?: unknown };
  if (typeof rec.type !== "string" || typeof rec.name !== "string") return undefined;
  return `${rec.type}\n${typeof rec.namespace === "string" ? rec.namespace : ""}\n${rec.name}`;
}

/**
 * Promote Codex Desktop's responses-lite `additional_tools` input items to top-level
 * `tools` and drop the items. The parser already collects these declarations into the
 * tool surface, but the outbound body keeps the item verbatim and Console Go's validator
 * rejects the unknown item type (`input[N] did not match any supported type`). Promoting
 * preserves every declaration (deduplicated by type/namespace/name) in the standard shape
 * the downstream namespace/custom lowering passes already handle.
 */
export function normalizeOpenCodeGoAdditionalTools(body: unknown): unknown {
  if (!body || typeof body !== "object" || Array.isArray(body)) return body;
  const record = body as Record<string, unknown>;
  if (!Array.isArray(record.input)) return body;
  const existing = Array.isArray(record.tools) ? (record.tools as unknown[]) : [];
  const seen = new Set<string>();
  for (const tool of existing) {
    const key = toolIdentityKey(tool);
    if (key !== undefined) seen.add(key);
  }
  const promoted: unknown[] = [];
  let changed = false;
  const input: unknown[] = [];
  for (const item of record.input as unknown[]) {
    if (!item || typeof item !== "object" || Array.isArray(item)
      || (item as { type?: unknown }).type !== "additional_tools"
      || !Array.isArray((item as { tools?: unknown }).tools)) {
      input.push(item);
      continue;
    }
    changed = true;
    for (const tool of (item as { tools: unknown[] }).tools) {
      const key = toolIdentityKey(tool);
      if (key === undefined || !seen.has(key)) {
        if (key !== undefined) seen.add(key);
        promoted.push(tool);
      }
    }
  }
  if (!changed) return body;
  return { ...record, input, tools: [...existing, ...promoted] };
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
