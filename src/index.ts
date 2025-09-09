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
  text: string;
  index: number;
  logprobs: null;
  finish_reason: string | null;
}

export interface LegacyCompletionUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

export interface LegacyCompletionResponse {
  id: string;
  object: "text_completion";
  created: number;
  model: string;
  choices: LegacyCompletionChoice[];
  usage?: LegacyCompletionUsage;
}

export interface LegacyCompletionStreamChunk {
  id?: string;
  object: "text_completion.chunk";
  choices: Array<{
    text: string;
    index: number;
    finish_reason: string | null;
  }>;
}

export type LegacyCompletionStreamFinal = LegacyCompletionResponse

export interface CompletionAdapterOptions {
  multiStrategy?: "parallel" | "prompt_split" | "first_only";
  promptSplitDelimiter?: string;
  promptSplitInstruction?: string;
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
    // user
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
    metadata: {
      // compat_source: "legacy_completion",
      // user
    }
  };

  Object.keys(req).forEach(k => {
    if (req[k] === undefined) delete req[k];
  });
  if (req.metadata && Object.keys(req.metadata).every(k => req.metadata[k] == null)) {
    delete req.metadata;
  }
  return req;
}

function responsesToCompletion(responsesObj: any): LegacyCompletionResponse {
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
      : (responsesObj.status_details && responsesObj.status_details.type) || "stop";

  return {
    id: responsesObj.id,
    object: "text_completion",
    created: responsesObj.created || Math.floor(Date.now() / 1000),
    model: responsesObj.model,
    choices: [
      {
        text: allText,
        index: 0,
        logprobs: null,
        finish_reason: finishReason
      }
    ],
    usage: {
      prompt_tokens: responsesObj.usage?.input_tokens,
      completion_tokens: responsesObj.usage?.output_tokens,
      total_tokens: responsesObj.usage?.total_tokens
    }
  };
}

function sum(nums: Array<number | undefined | null>): number {
  return nums.reduce((acc, v) => (acc as number) + (typeof v === "number" ? v : 0), 0) as number;
}

//////////////////// 主类 ////////////////////

export class OpenAICompletionCompat {
  private opts: Required<CompletionAdapterOptions>;

  constructor(private client: OpenAI, options?: {
    adapter?: CompletionAdapterOptions;
  }) {
    this.opts = {
      multiStrategy: options?.adapter?.multiStrategy || "first_only",
      promptSplitDelimiter: options?.adapter?.promptSplitDelimiter || "\n",
      promptSplitInstruction:
        options?.adapter?.promptSplitInstruction || "请生成 {n} 条结果，每条独立一行，不要额外解释。",
      attachRawResponses: options?.adapter?.attachRawResponses || false
    };
  }

  // 非流式
  async createCompletion(
    legacyReq: LegacyCompletionRequest
  ): Promise<LegacyCompletionResponse & { rawResponses?: any | any[] }> {
    const n = legacyReq.n ?? 1;

    if (n === 1 || this.opts.multiStrategy === "first_only") {
      const req = completionRequestToResponses(legacyReq);
      const responsesResult = await this.client.responses.create(req);
      const legacy = responsesToCompletion(responsesResult);
      if (this.opts.attachRawResponses) {
        (legacy as any).rawResponses = responsesResult;
      }
      return legacy;
    }

    switch (this.opts.multiStrategy) {
      case "parallel":
        return this._multiParallel(legacyReq, n);
      case "prompt_split":
        return this._multiPromptSplit(legacyReq, n);
      default:
        {
          const req2 = completionRequestToResponses(legacyReq);
          const r2 = await this.client.responses.create(req2);
          const legacy2 = responsesToCompletion(r2);
          if (this.opts.attachRawResponses) {
            (legacy2 as any).rawResponses = r2;
          }
          return legacy2;
        }
    }
  }

  /**
   * 流式
   * 返回一个对象，包含 abort() 可快速取消（调用 controller.abort()）
   */
  async createCompletionStream(
    legacyReq: LegacyCompletionRequest,
    handlers: {
      onDelta?: (chunk: LegacyCompletionStreamChunk) => void;
      onDone?: (finalResp: LegacyCompletionStreamFinal) => void;
      onError?: (err: any) => void;
    }
  ): Promise<{ abort: () => Promise<void> | void }> {
    if (legacyReq.n && legacyReq.n > 1) {
      console.warn("流式暂不支持 n>1（你可以并行创建多条流）");
    }

    const req = completionRequestToResponses(legacyReq);

    // SDK streaming：client.responses.stream(...)
    let finalResponse: any = null;
    let fullText = "";

    const stream = await this.client.responses.stream({
      ...req
    });

    (async () => {
      try {
        for await (const event of stream) {
          switch (event.type) {
            case "response.output_text.delta": {
              const delta = event.delta || "";
              fullText += delta;
              handlers.onDelta?.({
                id: event.item_id,
                object: "text_completion.chunk",
                choices: [
                  {
                    text: delta,
                    index: 0,
                    finish_reason: null
                  }
                ]
              });
              break;
            }
            case "response.output_text.done":
              // 段落结束，不一定需要处理
              break;
            case "response.completed":
              finalResponse = event.response;
              break;
            case "response.failed":
              handlers.onError?.(event.response || event);
              break;
          }
        }

        // 流自然结束后
        if (finalResponse) {
          const legacy = responsesToCompletion(finalResponse);
          legacy.choices[0].text = fullText; // 覆盖拼接
          handlers.onDone?.(legacy as LegacyCompletionStreamFinal);
        } else {
          // 未拿到 completed（可能被 abort）
          if (fullText.length > 0) {
            handlers.onDone?.({
              id: "unknown",
              object: "text_completion",
              created: Math.floor(Date.now() / 1000),
              model: legacyReq.model,
              choices: [{
                text: fullText,
                index: 0,
                logprobs: null,
                finish_reason: "stop"
              }],
              usage: {}
            });
          }
        }
      } catch (err) {
        handlers.onError?.(err);
      }
    })();

    return {
      abort: () => {
        try {
          stream.abort(); // SDK 提供的 abort
        } catch {
          // ignore
        }
      }
    };
  }

  //////////////////// 内部多输出策略 ////////////////////

  private async _multiParallel(
    legacyReq: LegacyCompletionRequest,
    n: number
  ): Promise<LegacyCompletionResponse & { rawResponses?: any[] }> {
    const singleReq = { ...legacyReq, n: 1 };
    const reqBody = completionRequestToResponses(singleReq);
    const tasks = Array.from({ length: n }, () =>
      this.client.responses.create(reqBody)
    );
    const results = await Promise.all(tasks);
    const legacyResults = results.map(r => responsesToCompletion(r));

    const merged: LegacyCompletionResponse = {
      id: legacyResults[0].id,
      object: "text_completion",
      created: legacyResults[0].created,
      model: legacyResults[0].model,
      choices: legacyResults.map((c, idx) => ({
        ...c.choices[0],
        index: idx
      })),
      usage: {
        prompt_tokens: sum(legacyResults.map(r => r.usage?.prompt_tokens)),
        completion_tokens: sum(legacyResults.map(r => r.usage?.completion_tokens)),
        total_tokens: sum(legacyResults.map(r => r.usage?.total_tokens))
      }
    };
    if (this.opts.attachRawResponses) {
      (merged as any).rawResponses = results;
    }
    return merged;
  }

  private async _multiPromptSplit(
    legacyReq: LegacyCompletionRequest,
    n: number
  ): Promise<LegacyCompletionResponse & { rawResponses?: any }> {
    const instruction = (this.opts.promptSplitInstruction || "").replace(/\{n\}/g, String(n));
    const delimiter = this.opts.promptSplitDelimiter || "\n";
    const combinedPrompt =
      `${legacyReq.messages.map(m => m.content).join(delimiter)}\n\n${instruction}\n（输出每条占一行，共 ${n} 行）`;

    const singleReq: LegacyCompletionRequest = {
      ...legacyReq,
      messages: [{ role: "user", content: combinedPrompt }],
      n: 1
    };

    const reqBody = completionRequestToResponses(singleReq);
    const responsesResult = await this.client.responses.create(reqBody);
    const legacySingle = responsesToCompletion(responsesResult);
    const rawText = legacySingle.choices[0].text.trim();

    let lines = rawText.split(delimiter).map(l => l.trim()).filter(Boolean);
    if (lines.length > n) lines = lines.slice(0, n);
    while (lines.length < n) lines.push("");

    const merged: LegacyCompletionResponse = {
      id: legacySingle.id,
      object: "text_completion",
      created: legacySingle.created,
      model: legacySingle.model,
      choices: lines.map((txt, idx) => ({
        text: txt,
        index: idx,
        logprobs: null,
        finish_reason: "stop"
      })),
      usage: legacySingle.usage
    };
    if (this.opts.attachRawResponses) {
      (merged as any).rawResponses = responsesResult;
    }
    return merged;
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
