import {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParams,
  ResponsesAPIRequest,
  ResponsesAPIResponse,
} from "./types";

export function chatCompletionToResponsesRequest(
  params: ChatCompletionCreateParams
): ResponsesAPIRequest {
  const {
    messages,
    model,
    max_tokens,
    temperature,
    top_p,
    stop,
    presence_penalty,
    frequency_penalty,
    user,
    tools,
    tool_choice,
  } = params;

  const req: ResponsesAPIRequest = {
    model,
    input: messages,
    max_output_tokens: max_tokens ?? undefined,
    temperature: temperature ?? undefined,
    top_p: top_p ?? undefined,
    stop: stop ?? undefined,
    presence_penalty: presence_penalty ?? undefined,
    frequency_penalty: frequency_penalty ?? undefined,
    tools: tools ?? undefined,
    tool_choice: tool_choice ?? undefined,
  };

  if (user) {
    req.metadata = { user };
  }

  return req;
}

export function responsesResponseToChatCompletion(
  response: ResponsesAPIResponse
): ChatCompletion {
  const output = response.output?.[0] ?? {};

  const combinedText = (output.content || [])
    .filter((c) => c.type === "output_text")
    .map((c) => c.text)
    .join("");

  const toolCalls = output.tool_calls;

  const finishReason =
    (response.status_details
      ?.type as ChatCompletion.Choice["finish_reason"]) ||
    (response.status === "completed" ? "stop" : "stop");

  const message: ChatCompletion.Choice["message"] = {
    role: "assistant",
    content: combinedText || null,
  };

  if (toolCalls && toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }

  return {
    id: response.id,
    object: "chat.completion",
    created: response.created || Math.floor(Date.now() / 1000),
    model: response.model,
    choices: [
      {
        index: 0,
        message: message,
        finish_reason: finishReason,
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: response.usage?.input_tokens ?? 0,
      completion_tokens: response.usage?.output_tokens ?? 0,
      total_tokens: response.usage?.total_tokens ?? 0,
    },
  };
}

export function constructTextChunk(
  id: string,
  model: string,
  delta: string,
  isFirst: boolean
): ChatCompletionChunk {
  return {
    id: id,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [
      {
        index: 0,
        delta: {
          ...(isFirst && { role: "assistant" }),
          content: delta || "",
        },
        finish_reason: null,
      },
    ],
  };
}

export function constructToolCallChunk(
  id: string,
  model: string,
  payload: any,
  isFirst: boolean
): ChatCompletionChunk {
  const { index, ...delta } = payload;
  return {
    id: id,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [
      {
        index: 0,
        delta: {
          ...(isFirst && { role: "assistant" }),
          content: null,
          tool_calls: [
            {
              index: index,
              ...delta,
            },
          ],
        },
        finish_reason: null,
      },
    ],
  };
}

export function constructFinalChunk(
  id: string,
  response: ResponsesAPIResponse
): ChatCompletionChunk {
  const finishReason =
    (response.status_details
      ?.type as ChatCompletion.Choice["finish_reason"]) ||
    (response.status === "completed" ? "stop" : "stop");

  return {
    id: id,
    object: "chat.completion.chunk",
    created: response.created || Math.floor(Date.now() / 1000),
    model: response.model,
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: response.usage.input_tokens,
      completion_tokens: response.usage.output_tokens,
      total_tokens: response.usage.total_tokens,
    },
  };
}