import 'dotenv/config';
import { HfInference } from '@huggingface/inference';
import { saveBlob } from './utils.js';

const hfi = new HfInference(process.env.HUGGINGFACEHUB_API_KEY);

const dish = `pizza`;

const result = await hfi.textToImage({
    model: `stabilityai/stable-diffusion-2`,
    inputs: dish,
    parameters: { negative_prompt: `blurry` },
});

await saveBlob(result, `output/basic-dish.jpg`);
