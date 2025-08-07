import { Buffer } from "node:buffer";

export default {
  async fetch(request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error(err);
      return new Response(err.message, fixCors({ status: err.status ?? 500 }));
    };
    try {
      const auth = request.headers.get("Authorization");
      const apiKey = auth?.split(" ")[1];
      const assert = (success) => {
        if (!success) {
          throw new HttpError(
            "The specified HTTP method is not allowed for the requested resource",
            400
          );
        }
      };
      const { pathname } = new URL(request.url);
      switch (true) {
        case pathname.endsWith("/chat/completions"):
          assert(request.method === "POST");
          return handleCompletions(await request.json(), apiKey).catch(
            errHandler
          );
        case pathname.endsWith("/embeddings"):
          assert(request.method === "POST");
          return handleEmbeddings(await request.json(), apiKey).catch(
            errHandler
          );
        case pathname.endsWith("/models"):
          assert(request.method === "GET");
          return handleModels(apiKey).catch(errHandler);
        default:
          throw new HttpError("404 Not Found", 404);
      }
    } catch (err) {
      return errHandler(err);
    }
  },
};

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  return { headers, status, statusText };
};

const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Headers": "*",
    },
  });
};

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";

// https://github.com/google-gemini/generative-ai-js/blob/cf223ff4a1ee5a2d944c53cddb8976136382bee6/src/requests/request.ts#L71
const API_CLIENT = "genai-js/0.21.0"; // npm view @google/generative-ai version
const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more,
});

async function handleModels(apiKey) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let { body } = response;
  if (response.ok) {
    const { models } = JSON.parse(await response.text());
    body = JSON.stringify(
      {
        object: "list",
        data: models.map(({ name }) => ({
          id: name.replace("models/", ""),
          object: "model",
          created: 0,
          owned_by: "",
        })),
      },
      null,
      "  "
    );
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings(req, apiKey) {
  if (typeof req.model !== "string") {
    throw new HttpError("model is not specified", 400);
  }
  if (!Array.isArray(req.input)) {
    req.input = [req.input];
  }
  let model;
  if (req.model.startsWith("models/")) {
    model = req.model;
  } else {
    req.model = DEFAULT_EMBEDDINGS_MODEL;
    model = "models/" + req.model;
  }
  const response = await fetch(
    `${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`,
    {
      method: "POST",
      headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
      body: JSON.stringify({
        requests: req.input.map((text) => ({
          model,
          content: { parts: { text } },
          outputDimensionality: req.dimensions,
        })),
      }),
    }
  );
  let { body } = response;
  if (response.ok) {
    const { embeddings } = JSON.parse(await response.text());
    body = JSON.stringify(
      {
        object: "list",
        data: embeddings.map(({ values }, index) => ({
          object: "embedding",
          index,
          embedding: values,
        })),
        model: req.model,
      },
      null,
      "  "
    );
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_MODEL = "gemini-1.5-pro-latest";
async function handleCompletions(req, apiKey) {
  let model = DEFAULT_MODEL;
  switch (true) {
    case typeof req.model !== "string":
      break;
    case req.model.startsWith("models/"):
      model = req.model.substring(7);
      break;
    case req.model.startsWith("gemini-"):
    case req.model.startsWith("learnlm-"):
      model = req.model;
  }
  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) {
    url += "?alt=sse";
  }

  const transformedRequest = await transformRequest(req);

  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(transformedRequest), // try
  });

  if (!response.ok) {
    const errorText = await response.text();
    return new Response(errorText, fixCors(response));
  }

  let body = response.body;
  if (response.ok) {
    let id = generateChatcmplId(); //"chatcmpl-8pMMaqXMK68B3nyDBrapTDrhkHBQK";
    if (req.stream) {
      body = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(
          new TransformStream({
            transform: parseStream,
            flush: parseStreamFlush,
            buffer: "",
          })
        )
        .pipeThrough(
          new TransformStream({
            transform: toOpenAiStream,
            flush: toOpenAiStreamFlush,
            streamIncludeUsage: req.stream_options?.include_usage,
            model,
            id,
            last: [],
          })
        )
        .pipeThrough(new TextEncoderStream());
    } else {
      body = await response.text();
      const data = JSON.parse(body);

      // Debug: Print raw API response to check for thinking content
      // console.log("Raw Gemini API response:", JSON.stringify(data, null, 2));

      body = processCompletionsResponse(data, model, id);
    }
  }
  return new Response(body, fixCors(response));
}

const harmCategory = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY",
];
const safetySettings = harmCategory.map((category) => ({
  category,
  threshold: "BLOCK_NONE",
}));
const fieldsMap = {
  stop: "stopSequences",
  n: "candidateCount", // not for streaming
  max_tokens: "maxOutputTokens",
  max_completion_tokens: "maxOutputTokens",
  temperature: "temperature",
  top_p: "topP",
  top_k: "topK", // non-standard
  frequency_penalty: "frequencyPenalty",
  presence_penalty: "presencePenalty",
  reasoning_effort: "reasoningEffort", // Gemini thinking support
};
const transformConfig = (req) => {
  let cfg = {};
  //if (typeof req.stop === "string") { req.stop = [req.stop]; } // no need
  for (let key in req) {
    const matchedKey = fieldsMap[key];
    if (matchedKey) {
      cfg[matchedKey] = req[key];
    }
  }

  // Handle thinking parameter for Gemini thinking models
  // Note: Thinking cannot be disabled on Gemini 2.5 Pro (always enabled)
  // For Gemini 2.5 Flash/Flash-Lite, set thinkingBudget to 0 to disable
  if (req.thinking) {
    if (req.thinking.type === "enabled" && req.thinking.budget_tokens) {
      cfg.thinkingConfig = {
        thinkingBudget: req.thinking.budget_tokens,
        includeThoughts: true,
      };
    } else if (
      typeof req.thinking === "object" &&
      req.thinking.thinkingBudget !== undefined
    ) {
      cfg.thinkingConfig = {
        thinkingBudget: req.thinking.thinkingBudget,
        includeThoughts: true,
      };
    }
  } else {
    // Set default thinking configuration if not specified，set max as default
    cfg.thinkingConfig = {
      thinkingBudget: 32768,
      includeThoughts: true,
    };
  }

  if (req.response_format) {
    switch (req.response_format.type) {
      case "json_schema":
        cfg.responseSchema = req.response_format.json_schema?.schema;
        if (cfg.responseSchema && "enum" in cfg.responseSchema) {
          cfg.responseMimeType = "text/x.enum";
          break;
        }
      // eslint-disable-next-line no-fallthrough
      case "json_object":
        cfg.responseMimeType = "application/json";
        break;
      case "text":
        cfg.responseMimeType = "text/plain";
        break;
      default:
        throw new HttpError("Unsupported response_format.type", 400);
    }
  }
  return cfg;
};

const parseImg = async (url) => {
  let mimeType, data;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText} (${url})`);
      }
      mimeType = response.headers.get("content-type");
      data = Buffer.from(await response.arrayBuffer()).toString("base64");
    } catch (err) {
      throw new Error("Error fetching image: " + err.toString());
    }
  } else {
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match) {
      throw new Error("Invalid image data: " + url);
    }
    ({ mimeType, data } = match.groups);
  }
  return {
    inlineData: {
      mimeType,
      data,
    },
  };
};

const transformMsg = async ({ role, content }) => {
  const parts = [];
  if (!Array.isArray(content)) {
    // system, user: string
    // assistant: string or null (Required unless tool_calls is specified.)
    parts.push({ text: content });
    return { role, parts };
  }
  // user:
  // An array of content parts with a defined type.
  // Supported options differ based on the model being used to generate the response.
  // Can contain text, image, or audio inputs.
  for (const item of content) {
    switch (item.type) {
      case "text":
        parts.push({ text: item.text });
        break;
      case "image_url":
        parts.push(await parseImg(item.image_url.url));
        break;
      case "input_audio":
        parts.push({
          inlineData: {
            mimeType: "audio/" + item.input_audio.format,
            data: item.input_audio.data,
          },
        });
        break;
      default:
        throw new TypeError(`Unknown "content" item type: "${item.type}"`);
    }
  }
  if (content.every((item) => item.type === "image_url")) {
    parts.push({ text: "" }); // to avoid "Unable to submit request because it must have a text parameter"
  }
  return { role, parts };
};

const transformToolResponseMsg = async (toolMsg) => {
  // Transform OpenAI tool response format to Gemini native format
  // OpenAI format: { role: "tool", tool_call_id: "call_xxx", content: "response" }
  // Gemini format: { role: "user", parts: [{ functionResponse: { name: "function_name", response: {...} } }] }

  const toolCallId = toolMsg.tool_call_id || "unknown_call";
  const content = toolMsg.content;

  // Use the function name provided by the backend, or fall back to extracting from call ID
  let functionName = toolMsg.name || "tool_response";
  if (!toolMsg.name && toolCallId && typeof toolCallId === "string") {
    // Try to extract meaningful name from call ID if possible
    const parts = toolCallId.split("_");
    if (parts.length > 1) {
      functionName = parts.slice(1).join("_") || "tool_response";
    }
  }

  // Parse the content to get the actual tool response
  let response = {};
  try {
    if (typeof content === "string") {
      // Try to parse as JSON first
      try {
        response = JSON.parse(content);
        // If it's a string that was JSON parsed, wrap it properly
        if (typeof response === "string") {
          response = { result: response };
        }
      } catch {
        // If not JSON, wrap the string content
        response = { result: content };
      }
    } else if (content && typeof content === "object") {
      response = content;
    } else {
      response = { result: String(content || "") };
    }
  } catch (e) {
    console.error("Error parsing tool response content:", e);
    response = { result: String(content || "") };
  }

  return {
    role: "user",
    parts: [
      {
        functionResponse: {
          name: functionName,
          response: response,
        },
      },
    ],
  };
};

const transformAssistantMsg = async (assistantMsg) => {
  // Handle assistant messages with tool_calls
  const parts = [];

  // Add text content if present
  if (assistantMsg.content && assistantMsg.content.trim()) {
    parts.push({ text: assistantMsg.content });
  }

  // Convert tool_calls to Gemini format
  if (assistantMsg.tool_calls && assistantMsg.tool_calls.length > 0) {
    for (const toolCall of assistantMsg.tool_calls) {
      if (toolCall.type === "function") {
        const functionCall = {
          functionCall: {
            name: toolCall.function.name,
            args: {},
          },
        };

        // Parse arguments
        try {
          if (toolCall.function.arguments) {
            const args =
              typeof toolCall.function.arguments === "string"
                ? JSON.parse(toolCall.function.arguments)
                : toolCall.function.arguments;
            functionCall.functionCall.args = args;
          }
        } catch (e) {
          console.error("Error parsing tool call arguments:", e);
          // Use empty args if parsing fails
        }

        parts.push(functionCall);
      }
    }
  }

  // If no content and no tool calls, add empty text to avoid errors
  if (parts.length === 0) {
    parts.push({ text: "" });
  }

  return { role: "model", parts };
};

const transformMessages = async (messages) => {
  if (!messages) {
    return;
  }
  const contents = [];
  let system_instruction;
  for (const item of messages) {
    if (item.role === "system") {
      delete item.role;
      system_instruction = await transformMsg(item);
    } else if (item.role === "tool") {
      // Handle tool response messages in Gemini native format
      contents.push(await transformToolResponseMsg(item));
    } else if (item.role === "assistant") {
      // Handle assistant messages with potential tool_calls
      contents.push(await transformAssistantMsg(item));
    } else {
      // Handle user messages
      item.role = "user";
      contents.push(await transformMsg(item));
    }
  }
  if (system_instruction && contents.length === 0) {
    contents.push({ role: "model", parts: { text: " " } });
  }
  //console.info(JSON.stringify(contents, 2));
  return { system_instruction, contents };
};

const transformTools = (tools) => {
  if (!tools || !Array.isArray(tools)) {
    return undefined;
  }

  const function_declarations = tools
    .filter((tool) => tool.type === "function" && tool.function)
    .map((tool) => {
      const func = tool.function;
      const declaration = {
        name: func.name,
        description: func.description || "",
      };

      // 只有当函数有参数时才添加parameters字段
      if (
        func.parameters &&
        func.parameters.properties &&
        Object.keys(func.parameters.properties).length > 0
      ) {
        declaration.parameters = func.parameters;
      }

      return declaration;
    });

  return function_declarations.length > 0
    ? { function_declarations }
    : undefined;
};

const transformRequest = async (req) => ({
  ...(await transformMessages(req.messages)),
  safetySettings,
  generationConfig: transformConfig(req),
  ...(req.tools && { tools: transformTools(req.tools) }),
});

const generateChatcmplId = () => {
  const characters =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () =>
    characters[Math.floor(Math.random() * characters.length)];
  return "chatcmpl-" + Array.from({ length: 29 }, randomChar).join("");
};

const reasonsMap = {
  //https://ai.google.dev/api/rest/v1/GenerateContentResponse#finishreason
  //"FINISH_REASON_UNSPECIFIED": // Default value. This value is unused.
  STOP: "stop",
  MAX_TOKENS: "length",
  SAFETY: "content_filter",
  RECITATION: "content_filter",
  //"OTHER": "OTHER",
  // :"function_call",
};
const SEP = "\n\n|>";
const transformCandidates = (key, cand) => {
  const parts = cand.content?.parts || [];

  // Separate different types of content
  const textParts = parts.filter((p) => p.text && !p.thought);
  const thinkingParts = parts.filter((p) => p.text && p.thought);
  const functionCallParts = parts.filter((p) => p.functionCall);

  const result = {
    index: cand.index || 0, // 0-index is absent in new -002 models response
    [key]: {
      role: "assistant",
      content: textParts.map((p) => p.text).join(SEP) || null,
    },
    logprobs: null,
    finish_reason: reasonsMap[cand.finishReason] || cand.finishReason,
  };

  // Add thinking/reasoning content if present
  if (thinkingParts.length > 0) {
    result[key].reasoning = thinkingParts.map((p) => p.text).join(SEP);
  }

  // Handle function calls (convert from Google format to OpenAI format)
  if (functionCallParts.length > 0) {
    result[key].tool_calls = functionCallParts.map((part, index) => ({
      id: `call_${generateToolCallId()}`,
      type: "function",
      function: {
        name: part.functionCall.name,
        arguments: JSON.stringify(part.functionCall.args || {}),
      },
    }));

    // If there are tool calls, content should be null in OpenAI format
    if (result[key].tool_calls.length > 0 && !result[key].content) {
      result[key].content = null;
    }
  }

  return result;
};

const generateToolCallId = () => {
  const characters =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () =>
    characters[Math.floor(Math.random() * characters.length)];
  return Array.from({ length: 24 }, randomChar).join("");
};
const transformCandidatesMessage = transformCandidates.bind(null, "message");
const transformCandidatesDelta = transformCandidates.bind(null, "delta");

const transformUsage = (data) => ({
  completion_tokens: data.candidatesTokenCount,
  prompt_tokens: data.promptTokenCount,
  total_tokens: data.totalTokenCount,
});

const processCompletionsResponse = (data, model, id) => {
  return JSON.stringify({
    id,
    choices: data.candidates.map(transformCandidatesMessage),
    created: Math.floor(Date.now() / 1000),
    model,
    //system_fingerprint: "fp_69829325d0",
    object: "chat.completion",
    usage: transformUsage(data.usageMetadata),
  });
};

const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
async function parseStream(chunk, controller) {
  chunk = await chunk;
  if (!chunk) {
    return;
  }
  this.buffer += chunk;
  do {
    const match = this.buffer.match(responseLineRE);
    if (!match) {
      break;
    }
    controller.enqueue(match[1]);
    this.buffer = this.buffer.substring(match[0].length);
  } while (true); // eslint-disable-line no-constant-condition
}
async function parseStreamFlush(controller) {
  if (this.buffer) {
    console.error("Invalid data:", this.buffer);
    controller.enqueue(this.buffer);
  }
}

function transformResponseStream(data, stop, first) {
  const item = transformCandidatesDelta(data.candidates[0]);
  if (stop) {
    item.delta = {};
  } else {
    item.finish_reason = null;
  }
  if (first) {
    item.delta.content = "";
    // Initialize reasoning field if thinking is present
    if (data.candidates[0].content?.parts?.some((p) => p.thought)) {
      item.delta.reasoning = "";
    }
    // Initialize tool_calls field if function calls are present
    if (data.candidates[0].content?.parts?.some((p) => p.functionCall)) {
      item.delta.tool_calls = [];
    }
  } else {
    delete item.delta.role;
  }
  const output = {
    id: this.id,
    choices: [item],
    created: Math.floor(Date.now() / 1000),
    model: this.model,
    //system_fingerprint: "fp_69829325d0",
    object: "chat.completion.chunk",
  };
  if (data.usageMetadata && this.streamIncludeUsage) {
    output.usage = stop ? transformUsage(data.usageMetadata) : null;
  }
  return "data: " + JSON.stringify(output) + delimiter;
}
const delimiter = "\n\n";
async function toOpenAiStream(chunk, controller) {
  const transform = transformResponseStream.bind(this);
  const line = await chunk;
  if (!line) {
    return;
  }
  let data;
  try {
    data = JSON.parse(line);
  } catch (err) {
    console.error(line);
    console.error(err);
    const length = this.last.length || 1; // at least 1 error msg
    const candidates = Array.from({ length }, (_, index) => ({
      finishReason: "error",
      content: { parts: [{ text: err }] },
      index,
    }));
    data = { candidates };
  }
  const cand = data.candidates[0];
  console.assert(
    data.candidates.length === 1,
    "Unexpected candidates count: %d",
    data.candidates.length
  );
  cand.index = cand.index || 0; // absent in new -002 models response
  if (!this.last[cand.index]) {
    controller.enqueue(transform(data, false, "first"));
  }
  this.last[cand.index] = data;
  if (cand.content) {
    // prevent empty data (e.g. when MAX_TOKENS)
    controller.enqueue(transform(data));
  }
}
async function toOpenAiStreamFlush(controller) {
  const transform = transformResponseStream.bind(this);
  if (this.last.length > 0) {
    for (const data of this.last) {
      controller.enqueue(transform(data, "stop"));
    }
    controller.enqueue("data: [DONE]" + delimiter);
  }
}
