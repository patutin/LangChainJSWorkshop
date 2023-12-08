import 'dotenv/config';
import { OpenAI } from 'langchain/llms/openai';

const openAi = new OpenAI({ temperature: 1 });

const product = `flying cows`;

const response = await openAi.predict(`What is a good name for a company that makes ${product}? Be creative about it.`);

console.info(`[predict response]:`, response);
