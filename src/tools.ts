import 'dotenv/config';
import { HfInference } from '@huggingface/inference';
import { initializeAgentExecutorWithOptions } from 'langchain/agents';
import { OpenAI } from 'langchain/llms/openai';
import { DynamicTool } from 'langchain/tools';

const tools = [
    new DynamicTool({
        name: `dish-generator`,
        description: `generates a random dish of the day; doesn't accept any input`,
        func: () => new OpenAI({ temperature: 1 })
            .predict(`Come up with a random "dish of the day". Only respond with the result.`),
    }),
    new DynamicTool({
        name: `base64-image-generator`,
        description: `accepts a prompt as an input and returns a base64 encoded image`,
        func: (input) => new HfInference(process.env.HUGGINGFACEHUB_API_KEY)
            .textToImage({
                model: `stabilityai/stable-diffusion-2`,
                inputs: input,
                parameters: { negative_prompt: `blurry` },
            })
            .then((response) => response.text())
            .then((buffer) => Buffer.from(buffer).toString(`base64`)),
    }),
    // new DynamicTool({
    //     name: `html-generator`,
    //     description: `accepts a dish name and a base64 image, creates an html page`,
    //     func: (input: string) => ``,
    // }),
];

const openAi = new OpenAI({ temperature: 0 });
const executor = await initializeAgentExecutorWithOptions(tools, openAi, {
    agentType: `zero-shot-react-description`,
    verbose: true,
});

const result = await executor.invoke({});
console.log(result);
