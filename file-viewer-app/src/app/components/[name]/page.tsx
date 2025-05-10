import fs from 'fs';
import path from 'path';
import { notFound } from 'next/navigation';
import { formatComponentName } from '@/utils/componentLoader';

// Import a client component wrapper that will handle the dynamic imports
import ComponentDisplay from '../../../components/ComponentDisplay';

// Define the params type
interface ComponentPageParams {
  params: {
    name: string;
  };
}

// Function to get the component content
async function getComponentContent(name: string) {
  const componentsDir = path.join(process.cwd(), '..', 'components');

  try {
    // Check for TSX file
    const tsxPath = path.join(componentsDir, `${name}.tsx`);
    if (fs.existsSync(tsxPath)) {
      const content = fs.readFileSync(tsxPath, 'utf-8');
      return { content, type: 'tsx' as const };
    }

    // Check for HTML file
    const htmlPath = path.join(componentsDir, `${name}.html`);
    if (fs.existsSync(htmlPath)) {
      const content = fs.readFileSync(htmlPath, 'utf-8');
      return { content, type: 'html' as const };
    }

    // No matching file found
    return null;
  } catch (error) {
    console.error(`Error reading component ${name}:`, error);
    return null;
  }
}

export default async function ComponentPage({ params }: ComponentPageParams) {
  // Await params before accessing its properties as per Next.js error and previous fixes
  const resolvedParams = await params;
  const componentName = resolvedParams.name;

  const componentData = await getComponentContent(componentName);

  if (!componentData) {
    notFound();
  }

  const { content, type } = componentData;
  const displayName = formatComponentName(componentName);

  return (
    <div className="min-h-screen p-8">
      <header className="mb-8">
        <div className="flex items-center mb-4">
          <a
            href="/"
            className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 mr-2"
          >
            ← Back to Components
          </a>
        </div>
        <h1 className="text-3xl font-bold mb-2">{displayName}</h1>
        <div className="flex items-center">
          <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded">
            {type.toUpperCase()}
          </span>
        </div>
      </header>

      <div className="space-y-8">
        {/* Conditionally render Source Code section only for TSX files */}
        {type === 'tsx' && (
          <div className="border rounded-lg p-4 bg-gray-50 dark:bg-gray-800/50">
            <h2 className="text-xl font-semibold mb-3">Source Code</h2>
            <div className="p-4 bg-gray-900 text-white rounded-md overflow-auto text-sm font-mono max-h-[600px]">
              <pre><code>{content}</code></pre>
            </div>
          </div>
        )}

        <div>
          <h2 className="text-xl font-semibold mb-3">
            {type === 'html' ? 'Rendered Component' : 'Formatted Code Preview'}
          </h2>
          <div className="border rounded-lg p-4 bg-white dark:bg-gray-800 min-h-[300px]">
            <ComponentDisplay
              content={content}
              type={type}
              name={componentName}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
