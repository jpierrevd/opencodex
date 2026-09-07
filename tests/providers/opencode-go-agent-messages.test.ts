import { expect, test } from "bun:test";
import { createResponsesPassthroughAdapter } from "../../src/adapters/openai-responses";
import { isOpenCodeGo, normalizeOpenCodeGoAdditionalTools, normalizeOpenCodeGoAgentMessages } from "../../src/adapters/opencode-go";
import { parseRequest } from "../../src/responses/parser";
import { routeModel } from "../../src/router";
import { createTranslatorBudget } from "../../src/lib/translator-budget";
import type { OcxProviderConfig } from "../../src/types";

const base: OcxProviderConfig = { adapter: "openai-responses", baseUrl: "https://opencode.ai/zen/go/v1", authMode: "key", apiKey: "synthetic-key" };
const body = () => ({ model: "muse-spark-1.3-contributor", input: [{ type: "agent_message", id: "amsg_test", author: "/root/reader", recipient: "/root/checker", content: [{ type: "input_text", text: "Exact assignment\nwith lines." }] }], stream: true });

test("Responses converts plaintext task and peer messages without mutating replay or losing routing identities", async () => {
  const raw = body(); const original = structuredClone(raw); const budget = createTranslatorBudget();
  const request = await createResponsesPassthroughAdapter(base).buildRequest(parseRequest(raw), { headers: new Headers(), translatorBudget: budget });
  const sent = JSON.parse(request.body as string);
  expect(sent.input[0].type).toBe("message");
  expect(sent.input[0].role).toBe("user");
  expect(sent.input[0].content[0].text).toContain('"author":"/root/reader"');
  expect(sent.input[0].content[0].text).toContain('"recipient":"/root/checker"');
  expect(sent.input[0].content[1]).toEqual(raw.input[0]!.content[0]);
  expect(sent.input[0].id).toBeUndefined();
  expect(raw).toEqual(original);
  budget.dispose();
});

test("ciphertext and unknown content are never reclassified as plaintext", () => {
  for (const part of [{ type: "encrypted_content", encrypted_content: "opaque" }, { type: "future_type", text: "opaque" }]) {
    const raw = { input: [{ type: "agent_message", content: [part] }] };
    expect(normalizeOpenCodeGoAgentMessages(raw)).toBe(raw);
  }
});

test("mixed plaintext and inter-agent ciphertext converts carrying only plaintext", () => {
  const text = { type: "input_text", text: "Message Type: NEW_TASK\nPayload" };
  const raw = { input: [{ type: "agent_message", id: "amsg_mixed", author: "/root", recipient: "/root/worker", content: [text, { type: "encrypted_content" }] }] };
  const original = structuredClone(raw);
  const result = normalizeOpenCodeGoAgentMessages(raw) as typeof raw;
  expect(result).not.toBe(raw);
  expect(result.input[0]!.type).toBe("message");
  expect(result.input[0]!.role).toBe("user");
  expect(result.input[0]!.content[0].text).toContain('\"author\":\"/root\"');
  expect(result.input[0]!.content[1]).toBe(text);
  expect(result.input[0]!.content).toHaveLength(2);
  expect(raw).toEqual(original);
});

test("mixed image, text and unknown parts keep only wire-safe parts", () => {
  const image = { type: "input_image", image_url: "data:image/png;base64,AAAA", detail: "high" };
  const text = { type: "input_text", text: "Inspect image" };
  const raw = { input: [{ type: "agent_message", content: [text, image, { type: "future_type", text: "opaque" }] }] };
  const result = normalizeOpenCodeGoAgentMessages(raw) as typeof raw;
  expect(result.input[0]!.type).toBe("message");
  expect(result.input[0]!.content).toEqual([text, image]);
});

test("additional_tools items promote to top-level tools and leave input", () => {
  const exec = { type: "custom", name: "exec", description: "run" };
  const group = { type: "namespace", name: "functions", tools: [exec] };
  const raw = { input: [
    { type: "additional_tools", id: "at_1", role: "developer", tools: [group] },
    { type: "message", role: "user", content: [{ type: "input_text", text: "hi" }] },
  ] };
  const original = structuredClone(raw);
  const result = normalizeOpenCodeGoAdditionalTools(raw) as typeof raw & { tools: unknown[] };
  expect(result).not.toBe(raw);
  expect(result.input.map((i: { type: string }) => i.type)).toEqual(["message"]);
  expect(result.tools).toEqual([group]);
  expect(raw).toEqual(original);
});

test("additional_tools merge dedupes against existing top-level tools", () => {
  const exec = { type: "custom", name: "exec", description: "run" };
  const raw = {
    tools: [{ type: "custom", name: "exec", description: "run" }],
    input: [{ type: "additional_tools", tools: [exec, { type: "function", name: "search" }] }],
  };
  const result = normalizeOpenCodeGoAdditionalTools(raw) as typeof raw & { tools: unknown[] };
  expect(result.tools).toHaveLength(2);
  expect(result.tools[1]).toEqual({ type: "function", name: "search" });
  expect(result.input).toEqual([]);
});

test("bodies without additional_tools items keep their reference", () => {
  const raw = { input: [{ type: "message", role: "user", content: [] }] };
  expect(normalizeOpenCodeGoAdditionalTools(raw)).toBe(raw);
  expect(normalizeOpenCodeGoAdditionalTools(null)).toBe(null);
});

test("image parts stay intact beside the assignment", () => {
  const image = { type: "input_image", image_url: "data:image/png;base64,AAAA", detail: "high" };
  const raw = { input: [{ type: "agent_message", content: [{ type: "input_text", text: "Inspect image" }, image] }] };
  const result = normalizeOpenCodeGoAgentMessages(raw) as typeof raw;
  expect(result.input[0]!.content[1]).toBe(image);
});

test("native forward keeps agent_message and auth/session headers unchanged", async () => {
  const budget = createTranslatorBudget();
  const provider = { ...base, baseUrl: "https://chatgpt.com/backend-api/codex", authMode: "forward" as const };
  const request = await createResponsesPassthroughAdapter(provider).buildRequest(parseRequest(body()), { headers: new Headers({ "session-id": "native-id", authorization: "Bearer native-test" }), translatorBudget: budget });
  expect(JSON.parse(request.body as string).input[0].type).toBe("agent_message");
  expect(new Headers(request.headers).get("x-opencode-session")).toBeNull();
  expect(new Headers(request.headers).get("session-id")).toBe("native-id");
  expect(new Headers(request.headers).get("authorization")).toBe("Bearer native-test");
  budget.dispose();
});

test("other destinations do not get Go normalization or session identity", async () => {
  const budget = createTranslatorBudget();
  const request = await createResponsesPassthroughAdapter({ ...base, baseUrl: "https://example.test/v1" }).buildRequest(parseRequest(body()), { headers: new Headers({ "session-id": "child-id" }), translatorBudget: budget });
  expect(JSON.parse(request.body as string).input[0].type).toBe("agent_message");
  expect(new Headers(request.headers).get("x-opencode-session")).toBeNull();
  budget.dispose();
});

test("canonical Go forward auth preserves private agent messages and the raw replay body", async () => {
  const raw = body();
  const original = structuredClone(raw);
  const parsed = parseRequest(raw);
  const budget = createTranslatorBudget();
  try {
    const request = await createResponsesPassthroughAdapter({ ...base, authMode: "forward" }).buildRequest(parsed, {
      headers: new Headers(), translatorBudget: budget,
    });
    expect(request.url).toBe("https://opencode.ai/zen/go/v1/responses");
    expect(JSON.parse(request.body as string).input[0]).toMatchObject({
      type: "agent_message", author: "/root/reader", recipient: "/root/checker",
      content: original.input[0]!.content,
    });
    expect(parsed._rawBody).toBe(raw);
    expect(raw).toEqual(original);
  } finally {
    budget.dispose();
  }
});

test.each(["https://opencode.ai/zen/go/v1", "https://opencode.ai/zen/go/v1/"])(
  "a renamed provider at %s still converts plaintext agent messages",
  async baseUrl => {
    const raw = body();
    const original = structuredClone(raw);
    const route = routeModel({
      port: 0, defaultProvider: "my-go", providers: { "my-go": { ...base, baseUrl, models: [raw.model] } },
    }, `my-go/${raw.model}`);
    const parsed = parseRequest(raw);
    const budget = createTranslatorBudget();
    try {
      const request = await createResponsesPassthroughAdapter(route.provider).buildRequest(parsed, {
        headers: new Headers(), translatorBudget: budget,
      });
      const sent = JSON.parse(request.body as string);
      expect(request.url).toBe("https://opencode.ai/zen/go/v1/responses");
      expect(sent.input[0]).toMatchObject({ type: "message", role: "user" });
      expect(sent.input[0].content.slice(1)).toEqual(original.input[0]!.content);
      expect(parsed._rawBody).toBe(raw);
      expect(raw).toEqual(original);
    } finally {
      budget.dispose();
    }
  },
);

test.each([
  "https://opencode.ai.evil.test/zen/go/v1",
  "http://opencode.ai/zen/go/v1",
  "https://opencode.ai/zen/v1",
  "https://opencode.ai/zen/go/v10",
])("Go-like destination %s preserves private agent messages", async baseUrl => {
  const raw = body();
  const original = structuredClone(raw);
  const parsed = parseRequest(raw);
  const budget = createTranslatorBudget();
  try {
    const request = await createResponsesPassthroughAdapter({ ...base, baseUrl }).buildRequest(parsed, {
      headers: new Headers(), translatorBudget: budget,
    });
    expect(JSON.parse(request.body as string).input[0]).toMatchObject({
      type: "agent_message", content: original.input[0]!.content,
    });
    expect(parsed._rawBody).toBe(raw);
    expect(raw).toEqual(original);
  } finally {
    budget.dispose();
  }
});

test.each(["not a URL", "https://", "/zen/go/v1"])(
  "malformed destination %s is not classified as Go",
  baseUrl => expect(isOpenCodeGo(baseUrl)).toBe(false),
);

test("Go conversion preserves file payloads beside text without mutating raw replay", async () => {
  const file = { type: "input_file", filename: "assignment.txt", file_data: "data:text/plain;base64,SGVsbG8=" };
  const message = body().input[0]!;
  const raw = { ...body(), input: [{ ...message, content: [...message.content, file] }] };
  const original = structuredClone(raw);
  const parsed = parseRequest(raw);
  const budget = createTranslatorBudget();
  try {
    const request = await createResponsesPassthroughAdapter(base).buildRequest(parsed, {
      headers: new Headers(), translatorBudget: budget,
    });
    const sent = JSON.parse(request.body as string);
    expect(sent.input[0]).toMatchObject({ type: "message", role: "user" });
    expect(sent.input[0].content.slice(1)).toEqual(original.input[0]!.content);
    expect(parsed._rawBody).toBe(raw);
    expect(raw).toEqual(original);
  } finally {
    budget.dispose();
  }
});

for (const { name, content } of [
  { name: "empty content", content: [] },
]) test(`Go preserves ${name} without partially converting it`, async () => {
  const raw = { ...body(), input: [{ ...body().input[0]!, content }] };
  const original = structuredClone(raw);
  expect(normalizeOpenCodeGoAgentMessages(raw)).toBe(raw);
  const parsed = parseRequest(raw);
  const budget = createTranslatorBudget();
  try {
    const request = await createResponsesPassthroughAdapter(base).buildRequest(parsed, {
      headers: new Headers(), translatorBudget: budget,
    });
    expect(JSON.parse(request.body as string).input[0]).toMatchObject({ type: "agent_message", content });
    expect(parsed._rawBody).toBe(raw);
    expect(raw).toEqual(original);
  } finally {
    budget.dispose();
  }
});

for (const { name, content, surviving } of [
  { name: "text mixed with an unknown part", content: [
    { type: "input_text", text: "Known prefix" }, { type: "future_type", text: "Do not lose this" },
  ], surviving: [{ type: "input_text", text: "Known prefix" }] },
  { name: "text mixed with ciphertext", content: [
    { type: "input_text", text: "Routing header" }, { type: "encrypted_content", encrypted_content: "opaque" },
  ], surviving: [{ type: "input_text", text: "Routing header" }] },
]) test(`Go converts ${name} carrying only wire-safe parts`, async () => {
  const raw = { ...body(), input: [{ ...body().input[0]!, content }] };
  const original = structuredClone(raw);
  const parsed = parseRequest(raw);
  const budget = createTranslatorBudget();
  try {
    const request = await createResponsesPassthroughAdapter(base).buildRequest(parsed, {
      headers: new Headers(), translatorBudget: budget,
    });
    const sent = JSON.parse(request.body as string).input[0];
    expect(sent.type).toBe("message");
    expect(sent.role).toBe("user");
    expect(sent.content.slice(1)).toEqual(surviving);
    expect(parsed._rawBody).toBe(raw);
    expect(raw).toEqual(original);
  } finally {
    budget.dispose();
  }
});
