import { expect, test } from 'vitest';
import OpenAI from 'openai';
import { OpenAICompletionCompat } from '../src/index';
import 'dotenv/config';

const messages: OpenAI.ChatCompletionMessageParam[] = [{ role: "system", content: "请你扮演一位吹牛大王，对用户的问题都夸大回复" }, { role: "user", content: "你觉得我美吗？" }];
const model = "doubao-seed-1-6-250615";
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: 'https://ark.cn-beijing.volces.com/api/v3',
});

test('createCompletion', async () => {
  const respRaw = await client.chat.completions.create({
    model,
    messages: messages,
    // @ts-expect-error
    thinking: {
      type: 'disabled',
    },
  });
  console.dir(respRaw, {
    depth: 10,
  });

  // const compat = new OpenAICompletionCompat(client);
  // const respSystem = await compat.createCompletion({
  //   model,
  //   messages: messages.slice(0, 1),
  //   // @ts-expect-error
  //   thinking: {
  //     type: 'disabled',
  //   },
  //   caching: {
  //     type: 'enabled',
  //   },
  // });
  // const resp = await compat.createCompletion({
  //   model,
  //   messages: messages.slice(1),
  //   // @ts-expect-error
  //   thinking: {
  //     type: 'disabled',
  //   },
  //   previous_response_id: respSystem.id,
  // });
  // expect(resp.usage?.prompt_tokens_details?.cached_tokens).toBeGreaterThan(0);
}, 60 * 1000);

test.only('createCompletionStream', async () => {
  // const respRaw = await client.responses.create({
  //   model,
  //   input: messages,
  //   // @ts-expect-error
  //   thinking: {
  //     type: 'disabled',
  //   },
  //   stream: true,
  //   max_output_tokens: 1,
  // })
  // const transformAsyncIterator = (async function* () {
  //   for await (const part of respRaw) {
  //     yield part;
  //   }
  // })();
  // for await (const part of transformAsyncIterator) {
  //   console.dir(part, {
  //     depth: 10,
  //   });
  // }
  const compat = new OpenAICompletionCompat(client);
  const respSystem = await compat.createCompletionStream({
    model,
    messages: messages.slice(0, 1),
    // @ts-expect-error
    thinking: {
      type: 'disabled',
    },
    caching: {
      type: 'enabled',
    },
  });
  for await (const part of respSystem) {
    console.dir(part, {
      depth: 10,
    });
  }
  // const resp = await compat.createCompletion({
  //   model,
  //   messages: messages.slice(1),
  //   // @ts-expect-error
  //   thinking: {
  //     type: 'disabled',
  //   },
  //   previous_response_id: respSystem.id,
  // });
  // console.dir(resp, {
  //   depth: 10,
  // }); 
}, 60 * 1000);
