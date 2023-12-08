import 'dotenv/config';
import { HfInference } from '@huggingface/inference';
import { ConsoleCallbackHandler } from 'langchain/callbacks';
import { OpenAI } from 'langchain/llms/openai';
import { PromptTemplate } from 'langchain/prompts';
import { StringOutputParser } from 'langchain/schema/output_parser';
import { RunnableSequence } from 'langchain/schema/runnable';
import { saveBlob } from './utils.js';

const openAi = new OpenAI({ temperature: 0.9 });
const hfi = new HfInference(process.env.HUGGINGFACEHUB_API_KEY);

const promptServing = PromptTemplate.fromTemplate(
    `How is this dish commonly garnished: {dish}? Only respond with the result.`,
);

const chainServing = promptServing.pipe(openAi).pipe(new StringOutputParser());

const combinedChain = RunnableSequence.from([
    {
        dish: (input) => input.dish,
        serving: chainServing,
    },
    (stepInput: { dish: string, serving: string }) => hfi.textToImage({
        model: `stabilityai/stable-diffusion-2`,
        inputs: `${ stepInput.dish.trim() }, ${ stepInput.serving.trim() }`,
        parameters: { negative_prompt: `blurry` },
    }),
]);

const dish = `beef wellington`;

const result = await combinedChain.invoke({ dish: dish }, { callbacks: [new ConsoleCallbackHandler()] });

await saveBlob(result, `output/served-dish.jpg`);
