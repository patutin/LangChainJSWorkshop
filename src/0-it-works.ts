import 'dotenv/config';
import { OpenAI } from 'langchain/llms/openai';

const openAi = new OpenAI({});

async function checkCall(): Promise<boolean> {
    try {
        const response = await openAi.call(`Say hello!`);
        console.info(`[call response]:`, response);
        return true;
    } catch (err) {
        console.error(`[call error]:`, err);
        return false;
    }
}

async function checkPredict(product: string): Promise<void> {
    try {
        const response = await openAi.predict(`What is a good name for a company that makes ${product}?`);
        console.info(`[predict response]:`, response);
    } catch (err) {
        console.error(`[predict error]:`, err);
    }
}

const helloSucceeded = await checkCall();
if (helloSucceeded) {
    await checkPredict(`flying cows`);
}
