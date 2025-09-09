import { test } from 'vitest';
import OpenAI from 'openai';
import { OpenAICompletionCompat } from '../src/index';
import 'dotenv/config';

test('createCompletion', async () => {
  const client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: 'https://ark.cn-beijing.volces.com/api/v3',
  });
  const respRaw = await client.chat.completions.create({
    model: "doubao-seed-1-6-250615",
    messages: [{ role: "user", content: "你好" }],
  });
  console.dir(respRaw); 

  const compat = new OpenAICompletionCompat(client);
  const resp = await compat.createCompletion({
    model: "doubao-seed-1-6-250615",
    messages: [{ role: "user", content: "你好" }],
    max_tokens: 100
  });
  console.dir(resp); 
});
