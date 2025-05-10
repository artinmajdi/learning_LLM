import fs from 'fs';
import path from 'path';
import { promisify } from 'util';

const readdir = promisify(fs.readdir);
const readFile = promisify(fs.readFile);
const stat = promisify(fs.stat);

export interface ComponentInfo {
  name: string;
  path: string;
  extension: string;
  content: string;
  type: 'tsx' | 'html';
}

export async function getComponentsList(componentsDir: string): Promise<ComponentInfo[]> {
  try {
    const files = await readdir(componentsDir);
    
    const componentPromises = files.map(async (file) => {
      const filePath = path.join(componentsDir, file);
      const stats = await stat(filePath);
      
      if (stats.isFile()) {
        const extension = path.extname(file).substring(1);
        
        if (extension === 'tsx' || extension === 'html') {
          const content = await readFile(filePath, 'utf-8');
          const name = path.basename(file, `.${extension}`);
          
          return {
            name,
            path: filePath,
            extension,
            content,
            type: extension as 'tsx' | 'html'
          };
        }
      }
      
      return null;
    });
    
    const components = await Promise.all(componentPromises);
    return components.filter((component): component is ComponentInfo => component !== null);
  } catch (error) {
    console.error('Error reading components directory:', error);
    return [];
  }
}

export function formatComponentName(name: string): string {
  return name
    .replace(/-/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}
