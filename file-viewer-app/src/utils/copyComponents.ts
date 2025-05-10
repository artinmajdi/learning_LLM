import fs from 'fs';
import path from 'path';
import { promisify } from 'util';

const readdir = promisify(fs.readdir);
const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);
const mkdir = promisify(fs.mkdir);
const stat = promisify(fs.stat);

/**
 * Copies components from the external components directory to the Next.js project
 * This allows us to properly import and render TSX components
 */
export async function copyComponentsToProject() {
  const sourceDir = path.join(process.cwd(), '..', 'components');
  const targetDir = path.join(process.cwd(), 'src', 'external-components');
  
  try {
    // Create target directory if it doesn't exist
    if (!fs.existsSync(targetDir)) {
      await mkdir(targetDir, { recursive: true });
    }
    
    // Read all files from the source directory
    const files = await readdir(sourceDir);
    
    // Filter for TSX and HTML files
    const componentFiles = files.filter(file => {
      const ext = path.extname(file).toLowerCase();
      return ext === '.tsx' || ext === '.html';
    });
    
    // Copy each file to the target directory
    for (const file of componentFiles) {
      const sourcePath = path.join(sourceDir, file);
      const targetPath = path.join(targetDir, file);
      
      // Check if it's a file
      const stats = await stat(sourcePath);
      if (stats.isFile()) {
        const content = await readFile(sourcePath, 'utf-8');
        
        // For TSX files, we need to ensure they're properly formatted for Next.js
        if (path.extname(file).toLowerCase() === '.tsx') {
          // Add export default to the component if it doesn't have one
          // This is a very simplified approach and may not work for all components
          let modifiedContent = content;
          
          if (!content.includes('export default')) {
            // Try to identify the component name and add export default
            const componentNameMatch = content.match(/function\s+(\w+)/);
            const componentName = componentNameMatch ? componentNameMatch[1] : null;
            
            if (componentName) {
              modifiedContent = `${content}\n\nexport default ${componentName};`;
            }
          }
          
          await writeFile(targetPath, modifiedContent, 'utf-8');
        } else {
          // For HTML files, just copy them as is
          await writeFile(targetPath, content, 'utf-8');
        }
        
        console.log(`Copied ${file} to ${targetPath}`);
      }
    }
    
    console.log('All components copied successfully');
    return componentFiles.map(file => path.basename(file, path.extname(file)));
  } catch (error) {
    console.error('Error copying components:', error);
    return [];
  }
}
