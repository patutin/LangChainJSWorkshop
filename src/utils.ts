import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join } from 'node:path';

export async function saveBlob(imageBlob: Blob, filePath: string) {
    const imageBuf = await imageBlob.arrayBuffer();
    const fullPath = isAbsolute(filePath) ? filePath : join(process.cwd(), filePath);
    await mkdir(dirname(fullPath), { recursive: true });
    await writeFile(fullPath, Buffer.from(imageBuf));
}
