import { HfInference } from '@huggingface/inference';
import 'dotenv/config';
import { OpenAI } from 'langchain/llms/openai';
import { PromptTemplate } from 'langchain/prompts';
import { StringOutputParser } from 'langchain/schema/output_parser';
import { RunnableSequence } from 'langchain/schema/runnable';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const openAi = new OpenAI({ temperature: 1 });
const hfi = new HfInference(process.env.HUGGINGFACEHUB_API_KEY);

const prompt = PromptTemplate.fromTemplate(
    `Translate this joke from {languageFrom} into {languageTo}: {joke}. Only respond with the result.`,
);

const chain = prompt.pipe(openAi).pipe(new StringOutputParser());

const combinedChain = RunnableSequence.from([
    chain,
    (imagePrompt: string) => hfi.textToImage({
        model: 'stabilityai/stable-diffusion-2',
        inputs: imagePrompt.trim(),
        parameters: { negative_prompt: 'blurry' },
    }),
]);

const imageBlob = await combinedChain.invoke({
    joke: 'Šta kaže Brus Vilis kad ode u prodavnicu kompjuterske opreme? "Daj hard."',
    languageFrom: 'Serbian',
    languageTo: 'Chinese',
}/*, { callbacks: [new ConsoleCallbackHandler()] }*/);

const imageBuf = await imageBlob.arrayBuffer();

const filePath = join(process.cwd(), performance.now() + '.jpg');
await writeFile(filePath, Buffer.from(imageBuf));
