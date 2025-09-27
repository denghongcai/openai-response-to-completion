import { OpenAIResponsesAdapter } from '../src/index';
import OpenAI from 'openai';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import type {
  ChatCompletion,
  ChatCompletionChunk,
} from 'openai/resources/chat/completions';

// Mock the OpenAI client
const mockOpenAIClient: any = {
  responses: {
    create: vi.fn(),
    stream: vi.fn(),
  },
};

describe('OpenAIResponsesAdapter', () => {
  let adapter: OpenAIResponsesAdapter;

  beforeEach(() => {
    vi.clearAllMocks();
    adapter = new OpenAIResponsesAdapter(mockOpenAIClient as OpenAI);
  });

  describe('Non-Streaming: chat.completions.create', () => {
    it('should handle a basic text request with n=1', async () => {
      const mockResponse = {
        id: 'res-123',
        created: 1677652288,
        model: 'gpt-4.1-mini',
        output: [{ content: [{ type: 'output_text', text: 'Hello, world!' }] }],
        status: 'completed',
        status_details: { type: 'stop' },
        usage: { input_tokens: 10, output_tokens: 20, total_tokens: 30 },
      };
      mockOpenAIClient.responses.create.mockResolvedValue(mockResponse);

      const result: ChatCompletion = await adapter.chat.completions.create({
        model: 'gpt-4.1-mini',
        messages: [{ role: 'user', content: 'Say hi' }],
      });

      expect(mockOpenAIClient.responses.create).toHaveBeenCalledOnce();
      expect(result.id).toBe('res-123');
      expect(result.object).toBe('chat.completion');
      expect(result.choices[0].message.content).toBe('Hello, world!');
      expect(result.choices[0].finish_reason).toBe('stop');
      expect(result.usage?.prompt_tokens).toBe(10);
    });

    it('should handle a tool call request', async () => {
      const mockResponse = {
        id: 'res-tool-123',
        created: 1677652300,
        model: 'gpt-4.1-mini',
        output: [
          {
            tool_calls: [
              {
                id: 'call_abc',
                type: 'function',
                function: {
                  name: 'get_weather',
                  arguments: '{"location":"Boston"}',
                },
              },
            ],
          },
        ],
        status: 'completed',
        status_details: { type: 'tool_calls' },
        usage: { input_tokens: 50, output_tokens: 20, total_tokens: 70 },
      };
      mockOpenAIClient.responses.create.mockResolvedValue(mockResponse);

      const result: ChatCompletion = await adapter.chat.completions.create({
        model: 'gpt-4.1-mini',
        messages: [{ role: 'user', content: "What's the weather in Boston?" }],
        tools: [
          {
            type: 'function',
            function: { name: 'get_weather', parameters: {} },
          },
        ],
      });

      expect(mockOpenAIClient.responses.create).toHaveBeenCalledOnce();
      expect(result.choices[0].finish_reason).toBe('tool_calls');
      expect(result.choices[0].message.content).toBe(null);
      expect(result.choices[0].message.tool_calls).toBeDefined();
      expect(result.choices[0].message.tool_calls?.[0].function.name).toBe(
        'get_weather',
      );
      expect(result.choices[0].message.tool_calls?.[0].function.arguments).toBe(
        '{"location":"Boston"}',
      );
    });

    it('should handle n > 1 using parallel strategy', async () => {
      const mockResponse1 = {
        id: 'res-1',
        created: 1,
        model: 'gpt-4.1-mini',
        output: [{ content: [{ type: 'output_text', text: 'Response 1' }] }],
        status: 'completed',
        status_details: { type: 'stop' },
        usage: { input_tokens: 10, output_tokens: 11, total_tokens: 21 },
      };
      const mockResponse2 = {
        id: 'res-2',
        created: 2,
        model: 'gpt-4.1-mini',
        output: [{ content: [{ type: 'output_text', text: 'Response 2' }] }],
        status: 'completed',
        status_details: { type: 'stop' },
        usage: { input_tokens: 10, output_tokens: 12, total_tokens: 22 },
      };
      mockOpenAIClient.responses.create
        .mockResolvedValueOnce(mockResponse1)
        .mockResolvedValueOnce(mockResponse2);

      const result: ChatCompletion = await adapter.chat.completions.create({
        model: 'gpt-4.1-mini',
        messages: [{ role: 'user', content: 'Give me two responses' }],
        n: 2,
      });

      expect(mockOpenAIClient.responses.create).toHaveBeenCalledTimes(2);
      expect(result.choices).toHaveLength(2);
      expect(result.choices[0].message.content).toBe('Response 1');
      expect(result.choices[0].index).toBe(0);
      expect(result.choices[1].message.content).toBe('Response 2');
      expect(result.choices[1].index).toBe(1);
      expect(result.usage?.prompt_tokens).toBe(20);
      expect(result.usage?.completion_tokens).toBe(23);
      expect(result.usage?.total_tokens).toBe(43);
    });
  });

  describe('Streaming: chat.completions.create({ stream: true })', () => {
    it('should yield ChatCompletionChunk objects for a text stream', async () => {
      async function* mockStream() {
        yield {
          type: 'response.output_text.delta',
          item_id: 'stream-123',
          delta: 'Hello',
        };
        yield {
          type: 'response.output_text.delta',
          item_id: 'stream-123',
          delta: ', ',
        };
        yield {
          type: 'response.output_text.delta',
          item_id: 'stream-123',
          delta: 'world!',
        };
        yield {
          type: 'response.completed',
          response: {
            id: 'stream-123-final',
            created: 1677652299,
            model: 'gpt-4.1-mini',
            status: 'completed',
            status_details: { type: 'stop' },
            usage: { input_tokens: 5, output_tokens: 15, total_tokens: 20 },
          },
        };
      }
      mockOpenAIClient.responses.stream.mockReturnValue(mockStream());

      const stream = await adapter.chat.completions.create({
        model: 'gpt-4.1-mini',
        messages: [{ role: 'user', content: 'Stream me a greeting' }],
        stream: true,
      });

      const chunks: ChatCompletionChunk[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(1);
      expect(chunks[0].choices[0].delta.role).toBe('assistant');
      expect(chunks[0].choices[0].delta.content).toBe('Hello');
      const finalChunk = chunks[chunks.length - 1];
      expect(finalChunk.choices[0].finish_reason).toBe('stop');
    });

    it('should yield ChatCompletionChunk objects for a tool call stream', async () => {
      async function* mockToolStream() {
        yield {
          type: 'response.output_tool_calls.delta',
          item_id: 'stream-tool-123',
          payload: {
            index: 0,
            id: 'call_abc',
            type: 'function',
            function: { name: 'get_weather' },
          },
        };
        yield {
          type: 'response.output_tool_calls.delta',
          item_id: 'stream-tool-123',
          payload: { index: 0, function: { arguments: '{"loc' } },
        };
        yield {
          type: 'response.output_tool_calls.delta',
          item_id: 'stream-tool-123',
          payload: { index: 0, function: { arguments: 'ation":"Boston"}' } },
        };
        yield {
          type: 'response.completed',
          response: {
            id: 'stream-tool-123-final',
            created: 1677652310,
            model: 'gpt-4.1-mini',
            status: 'completed',
            status_details: { type: 'tool_calls' },
            usage: { input_tokens: 50, output_tokens: 20, total_tokens: 70 },
          },
        };
      }
      mockOpenAIClient.responses.stream.mockReturnValue(mockToolStream());

      const stream = await adapter.chat.completions.create({
        model: 'gpt-4.1-mini',
        messages: [{ role: 'user', content: 'Weather in Boston?' }],
        tools: [
          {
            type: 'function',
            function: { name: 'get_weather', parameters: {} },
          },
        ],
        stream: true,
      });

      const chunks: ChatCompletionChunk[] = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(1);
      // First chunk should establish the tool call
      expect(chunks[0].choices[0].delta.role).toBe('assistant');
      expect(chunks[0].choices[0].delta.tool_calls?.[0].function?.name).toBe(
        'get_weather',
      );

      // Subsequent chunks should append arguments
      expect(
        chunks[1].choices[0].delta.tool_calls?.[0].function?.arguments,
      ).toBe('{"loc');
      expect(
        chunks[2].choices[0].delta.tool_calls?.[0].function?.arguments,
      ).toBe('ation":"Boston"}');

      // Final chunk should have the correct finish reason
      const finalChunk = chunks[chunks.length - 1];
      expect(finalChunk.choices[0].finish_reason).toBe('tool_calls');
    });

    it('should throw an error if n > 1 for streams', async () => {
      await expect(
        adapter.chat.completions.create({
          model: 'gpt-4.1-mini',
          messages: [{ role: 'user', content: 'This should fail' }],
          stream: true,
          n: 2,
        }),
      ).rejects.toThrow(
        'Streaming with `n > 1` is not supported in a single request.',
      );
    });
  });
});
