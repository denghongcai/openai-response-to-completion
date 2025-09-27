/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * openaiResponseAdapter.ts
 *
 * This file provides an adapter that mimics the official OpenAI Chat Completions API
 * but internally calls a non-standard `responses` API.
 * It is designed to be a drop-in replacement for the `openai` package's `chat.completions` object.
 */

import type OpenAI from "openai";
import {
  AdapterOptions,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParams,
  ResponsesAPIResponse,
} from "./types";
import {
  chatCompletionToResponsesRequest,
  constructFinalChunk,
  constructTextChunk,
  constructToolCallChunk,
  responsesResponseToChatCompletion,
} from "./mappers";

//////////////////// MAIN ADAPTER CLASS ////////////////////

export class OpenAIResponseAdapter {
  public chat: {
    completions: {
      create: (
        params: ChatCompletionCreateParams
      ) => Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>>;
    };
  };

  private opts: Required<AdapterOptions>;

  constructor(
    private client: OpenAI,
    options?: AdapterOptions
  ) {
    this.opts = {
      multiStrategy: options?.multiStrategy || "parallel",
      attachRawResponses: options?.attachRawResponses || false,
    };

    this.chat = {
      completions: {
        create: this.create.bind(this),
      },
    };
  }

  private create(
    params: ChatCompletionCreateParams
  ): Promise<ChatCompletion | AsyncIterable<ChatCompletionChunk>> {
    if (params.stream) {
      if (params.n && params.n > 1) {
        return Promise.reject(new Error("Streaming with `n > 1` is not supported in a single request."));
      }
      return Promise.resolve(this.createStream(params));
    }
    return this.createNonStream(params);
  }

  private async createNonStream(
    params: ChatCompletionCreateParams
  ): Promise<ChatCompletion> {
    const n = params.n ?? 1;

    if (n === 1) {
      const responsesReq = chatCompletionToResponsesRequest(params);
      const responsesResult = await (this.client as any).responses.create(responsesReq);
      const chatCompletion = responsesResponseToChatCompletion(responsesResult);
      if (this.opts.attachRawResponses) {
        (chatCompletion as any).rawResponse = responsesResult;
      }
      return chatCompletion;
    }

    if (this.opts.multiStrategy === "parallel") {
      const singleReqParams = { ...params, n: 1 };
      const responsesReq = chatCompletionToResponsesRequest(singleReqParams);

      const tasks = Array.from({ length: n }, () =>
        (this.client as any).responses.create(responsesReq)
      );

      const results: ResponsesAPIResponse[] = await Promise.all(tasks);
      const chatCompletions = results.map(r => responsesResponseToChatCompletion(r));

      const mergedCompletion: ChatCompletion = {
        id: chatCompletions[0].id,
        object: "chat.completion",
        created: chatCompletions[0].created,
        model: chatCompletions[0].model,
        choices: chatCompletions.flatMap((c, i) =>
          c.choices.map(choice => ({...choice, index: i}))
        ),
        usage: {
          prompt_tokens: chatCompletions.reduce((sum, c) => sum + (c.usage?.prompt_tokens ?? 0), 0),
          completion_tokens: chatCompletions.reduce((sum, c) => sum + (c.usage?.completion_tokens ?? 0), 0),
          total_tokens: chatCompletions.reduce((sum, c) => sum + (c.usage?.total_tokens ?? 0), 0),
        }
      };

      if (this.opts.attachRawResponses) {
        (mergedCompletion as any).rawResponses = results;
      }
      return mergedCompletion;
    }

    const responsesReq = chatCompletionToResponsesRequest(params);
    const responsesResult = await (this.client as any).responses.create(responsesReq);
    return responsesResponseToChatCompletion(responsesResult);
  }

  private async *createStream(
    params: ChatCompletionCreateParams
  ): AsyncIterable<ChatCompletionChunk> {
    const req = chatCompletionToResponsesRequest(params);
    const stream = await (this.client as any).responses.stream(req);

    let responseId: string | undefined;
    const model = params.model;
    let isFirstChunk = true;
    let finalResponse: ResponsesAPIResponse | null = null;

    for await (const event of stream) {
      responseId = responseId || event.item_id;

      switch (event.type) {
        case "response.output_text.delta": {
          const chunk = constructTextChunk(responseId!, model, event.delta, isFirstChunk);
          yield chunk;
          isFirstChunk = false;
          break;
        }
        case "response.output_tool_calls.delta": {
          const chunk = constructToolCallChunk(responseId!, model, event.payload, isFirstChunk);
          yield chunk;
          isFirstChunk = false;
          break;
        }
        case "response.completed":
          finalResponse = event.response;
          break;
        case "response.failed":
          throw new Error(`Stream failed: ${JSON.stringify(event.response || event)}`);
      }
    }

    if (finalResponse) {
      const finalChunk = constructFinalChunk(responseId!, finalResponse);
      yield finalChunk;
    }
  }
}