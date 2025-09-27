import type OpenAI from "openai";

// Official OpenAI types
export type ChatCompletionCreateParams = OpenAI.Chat.ChatCompletionCreateParams;
export type ChatCompletion = OpenAI.Chat.ChatCompletion;
export type ChatCompletionChunk = OpenAI.Chat.ChatCompletionChunk;
export type ChatCompletionTool = OpenAI.Chat.ChatCompletionTool;
export type ChatCompletionToolChoiceOption = OpenAI.Chat.ChatCompletionToolChoiceOption;
export type ChatCompletionMessageToolCall = OpenAI.Chat.ChatCompletionMessageToolCall;

/**
 * Represents the non-standard `responses` API request format.
 * This is an internal type.
 */
export interface ResponsesAPIRequest {
  model: string;
  input: OpenAI.Chat.ChatCompletionMessageParam[];
  max_output_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  metadata?: {
    user?: string;
  };
  // Support for Tool Calling
  tools?: ChatCompletionTool[];
  tool_choice?: ChatCompletionToolChoiceOption;
}

/**
 * Represents the non-standard `responses` API response object.
 * This is a simplified internal representation.
 */
export interface ResponsesAPIResponse {
  id: string;
  created: number;
  model: string;
  output: Array<{
    content?: Array<{
      type: "output_text";
      text: string;
    }>;
    // Support for Tool Calling
    tool_calls?: ChatCompletionMessageToolCall[];
  }>;
  status: string;
  status_details: {
    type: string; // e.g., 'stop', 'tool_calls'
  };
  usage: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  };
}

export interface AdapterOptions {
  multiStrategy?: "parallel";
  attachRawResponses?: boolean;
}