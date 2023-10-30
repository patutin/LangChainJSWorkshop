import { HfInference } from '@huggingface/inference';
import 'dotenv/config';
import { ConsoleCallbackHandler } from 'langchain/callbacks';
import { HuggingFaceInference } from 'langchain/llms/hf';
import { OpenAI } from 'langchain/llms/openai';
import { PromptTemplate } from 'langchain/prompts';
import { BytesOutputParser, StringOutputParser } from 'langchain/schema/output_parser';
import { RunnableSequence } from 'langchain/schema/runnable';
import { NodeFileStore } from 'langchain/stores/file/node';
import { WriteFileTool } from 'langchain/tools';
import { createWriteStream, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const openAi = new OpenAI({ temperature: 1 });
const hf = new HuggingFaceInference({ model: 'stabilityai/stable-diffusion-2' });
const hfi = new HfInference(process.env.HUGGINGFACEHUB_API_KEY);

const fileStore = new NodeFileStore(process.cwd());

const promptJ = PromptTemplate.fromTemplate(
    `Translate this joke from {languageFrom} into {languageTo}: {joke}. Only respond with the result.`,
);

const chainJ = promptJ.pipe(openAi).pipe(new StringOutputParser());

const combinedChainJ = RunnableSequence.from([
    chainJ,
    // (imagePrompt: string) => hf.call(imagePrompt),
    (imagePrompt: string) => hfi.textToImage({
        model: 'stabilityai/stable-diffusion-2',
        inputs: imagePrompt,
        parameters: { negative_prompt: 'blurry' },
    }),
    new BytesOutputParser(),
]);

const blob = await combinedChainJ.invoke({
    joke: 'Šta kaže Brus Vilis kad ode u prodavnicu kompjuterske opreme? "Daj hard."',
    languageFrom: 'Serbian',
    languageTo: 'Chinese',
}, { callbacks: [new ConsoleCallbackHandler()] });

const filePath = join(process.cwd(), performance.now() + '.jpg');
const file = createWriteStream(filePath);

// for await (const chunk of blob) {
//     if (chunk) chunk.pipe(file);
// }
// file.close();

file.write(blob, () => file.close())

// writeFileSync(filePath, Buffer.from(blob));
