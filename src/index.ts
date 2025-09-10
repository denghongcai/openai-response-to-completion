/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * openaiCompletionCompat.ts
 *
 * 用最新 Responses API 模拟旧 /v1/completions。
 * 仅使用官方 openai Node SDK（>= 4.x）。不直接使用 fetch。
 *
 * 安装：
 *   pnpm add openai
 *
 * 适配内容：
 * - prompt -> input
 * - max_tokens -> max_output_tokens
 * - usage 字段映射
 * - 流式事件 -> 旧 completions 增量回调
 * - n 多结果的三种策略：parallel / prompt_split / first_only
 *
 * 不支持/降级：
 * - logprobs -> 始终 null
 * - n>1 的真正并行流式（需自行多个流）
 */

import type OpenAI from "openai";

//////////////////// 类型定义 ////////////////////

export interface LegacyCompletionRequest {
  model: string;
  messages: OpenAI.ChatCompletionMessageParam[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  user?: string;
  n?: number;
  logprobs?: number;
}

export interface LegacyCompletionChoice {
  message?: OpenAI.ChatCompletionMessageParam;
  delta?: OpenAI.ChatCompletionMessageParam;
  index: number;
  logprobs: null;
  finish_reason: string | null;
}

export interface LegacyCompletionUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  completion_tokens_details?: {
    reasoning_tokens?: number;
  };
  prompt_tokens_details?: {
    cached_tokens?: number;
  };
}

export interface LegacyCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: LegacyCompletionChoice[];
  usage?: LegacyCompletionUsage;
}

export interface LegacyCompletionStreamChunk {
  id?: string;
  object: "chat.completion.chunk";
  choices: Array<{
    text: string;
    index: number;
    finish_reason: string | null;
  }>;
}

export type LegacyCompletionStreamFinal = LegacyCompletionResponse

export interface CompletionAdapterOptions {
  attachRawResponses?: boolean;
}

//////////////////// 辅助函数 ////////////////////

function completionRequestToResponses(legacyReq: LegacyCompletionRequest) {
  const {
    model,
    messages,
    max_tokens,
    temperature,
    top_p,
    stop,
    presence_penalty,
    frequency_penalty,
    ...rest
  } = legacyReq;

  const req: Record<string, any> = {
    model,
    input: messages,
    max_output_tokens: max_tokens,
    temperature,
    top_p,
    stop,
    presence_penalty,
    frequency_penalty,
    ...rest,
  };

  Object.keys(req).forEach(k => {
    if (req[k] === undefined) delete req[k];
  });
  if (req.metadata && Object.keys(req.metadata).every(k => req.metadata[k] == null)) {
    delete req.metadata;
  }
  return req;
}

function responsesToCompletion(responsesObj: OpenAI.Responses.Response): LegacyCompletionResponse {
  const allText = (responsesObj.output || [])
    .flatMap((o: any) =>
      (o.content || [])
        .filter((c: any) => c.type === "output_text")
        .map((c: any) => c.text)
    )
    .join("");

  const finishReason =
    responsesObj.status === "completed"
      ? "stop"
      : responsesObj.incomplete_details?.reason || "stop";

  console.log('>>>>>');
  console.dir(responsesObj, {
    depth: 10,
  });
  console.log('<<<<<');

  return {
    id: responsesObj.id,
    object: "chat.completion",
    created: responsesObj.created_at || Math.floor(Date.now() / 1000),
    model: responsesObj.model,
    choices: [
      {
        message: {
          role: (responsesObj.output?.[0] as any)?.role,
          content: allText,
        },
        index: 0,
        logprobs: null,
        finish_reason: finishReason
      }
    ],
    usage: {
      prompt_tokens: responsesObj.usage?.input_tokens,
      completion_tokens: responsesObj.usage?.output_tokens,
      total_tokens: responsesObj.usage?.total_tokens,
      prompt_tokens_details: responsesObj.usage?.input_tokens_details,
      completion_tokens_details: responsesObj.usage?.output_tokens_details,
    }
  };
}

//////////////////// 主类 ////////////////////

export class OpenAICompletionCompat {
  private opts: Required<CompletionAdapterOptions>;

  constructor(private client: OpenAI, options?: {
    adapter?: CompletionAdapterOptions;
  }) {
    this.opts = {
      attachRawResponses: options?.adapter?.attachRawResponses || false
    };
  }

  // 非流式
  async createCompletion(
    legacyReq: LegacyCompletionRequest,
    options?: OpenAI.RequestOptions,
  ): Promise<LegacyCompletionResponse & { rawResponses?: any | any[] }> {
    const req = completionRequestToResponses(legacyReq);
    const responsesResult = await this.client.responses.create(req, options);
    const legacy = responsesToCompletion(responsesResult);
    if (this.opts.attachRawResponses) {
      (legacy as any).rawResponses = responsesResult;
    }
    return legacy;
  }

  /**
   * 流式
   * 返回一个对象，包含 abort() 可快速取消（调用 controller.abort()）
   */
  async createCompletionStream(
    legacyReq: LegacyCompletionRequest,
    options?: OpenAI.RequestOptions,
  ): Promise<AsyncIterable<LegacyCompletionStreamChunk> & {
    controller: AbortController;
  }> {
    const req = completionRequestToResponses(legacyReq);

    // SDK streaming：client.responses.stream(...)
    let finalResponse: OpenAI.Responses.ResponseCompletedEvent['response'] | null = null;
    let fullText = "";
    let role = "";
    let type = "";

    const stream = await this.client.responses.create({
      ...req,
      stream: true,
    }, options);

    const transformAsyncIterator = (async function* () {
      for await (const event of stream) {
        switch (event.type) {
          case "response.output_item.added": {
            type = event.item.type;
            if (event.item.type === "message") {
              role = event.item.role;
            }
            break;
          }
          case "response.output_text.delta": {
            const delta = event.delta || "";
            fullText += delta;
            yield {
              id: event.item_id,
              object: "chat.completion.chunk",
              choices: [
                {
                  delta: {
                    content: delta,
                    role,
                    type,
                  },
                  index: 0,
                  finish_reason: null
                }
              ]
            };
            break;
          }
          case "response.output_text.done":
            // 段落结束，不一定需要处理
            break;
          case "response.completed":
            finalResponse = event.response;
            break;
          case "response.failed":
            yield event.response;
            return;
        }
      }
      // 流自然结束后
      if (finalResponse) {
        yield {
          ...responsesToCompletion(finalResponse),
          choices: [{
            delta: {
              content: '',
              role,
              type,
            },
            finish_reason: 'stop',
            index: 0,
          }],
        };
      } else {
        // 未拿到 completed（可能被 abort）
        if (fullText.length > 0) {
          yield {
            id: "unknown",
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: legacyReq.model,
            choices: [{
              delta: {
                content: fullText,
                role,
                type,
              },
              index: 0,
              logprobs: null,
              finish_reason: "stop"
            }],
            usage: {}
          };
        }
      }
    })();

    (transformAsyncIterator as any).controller = stream.controller;

    return transformAsyncIterator as unknown as (AsyncIterable<LegacyCompletionStreamChunk> & {
      controller: AbortController;
    });
  }


}

//////////////////// 使用示例 //////////////////////
/*
import { OpenAICompletionCompat } from "./openaiCompletionCompat";

const compat = new OpenAICompletionCompat(process.env.OPENAI_API_KEY!, {
  adapter: {
    multiStrategy: "parallel", // 或 "prompt_split" | "first_only"
    attachRawResponses: false
  }
});

(async () => {
  const r = await compat.createCompletion({
    model: "gpt-4.1-mini",
    prompt: "写一句激励程序员的话。",
    max_tokens: 60
  });
  console.log("单输出:", r.choices[0].text);

  const r2 = await compat.createCompletion({
    model: "gpt-4.1-mini",
    prompt: "给出不同风格的激励短句。",
    n: 3,
    max_tokens: 60
  });
  console.log("多输出:", r2.choices.map(c => c.text));

  const controller = await compat.createCompletionStream(
    {
      model: "gpt-4.1-mini",
      prompt: "实时逐步生成一首四行小诗：",
      max_tokens: 80
    },
    {
      onDelta: chunk => process.stdout.write(chunk.choices[0].text),
      onDone: finalR => {
        console.log("\n完成：", finalR.choices[0].text);
      },
      onError: err => console.error("流错误:", err)
    }
  );

  // 需要中途取消可：
  // setTimeout(() => controller.abort(), 1500);
})();
*/

//////////////////// 注意 //////////////////////
/**
 * 1. 依赖 SDK 的 streaming 实现，事件类型以当前版本为准：response.output_text.delta / response.completed / response.error。
 * 2. n>1 的并行流如果需要，请自行多次调用 createCompletionStream。
 * 3. prompt_split 策略质量不保证；生产环境建议 parallel。
 * 4. tokens 计费以 usage 字段合并或原始单次为准；parallel 会真实多次请求。
 * 5. 后续如果 Responses 原生支持多 completion 输出，可替换 multiStrategy 逻辑。
 */
