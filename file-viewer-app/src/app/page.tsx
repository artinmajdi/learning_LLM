import Link from "next/link";
import fs from "fs";
import path from "path";
import { formatComponentName } from "@/utils/componentLoader";

// Function to get component files from the external components directory
function getComponentFiles() {
  const componentsDir = path.join(process.cwd(), "..", "components");
  
  try {
    if (!fs.existsSync(componentsDir)) {
      return [];
    }
    
    const files = fs.readdirSync(componentsDir);
    
    return files
      .filter(file => {
        const ext = path.extname(file).toLowerCase();
        return ext === ".tsx" || ext === ".html";
      })
      .map(file => {
        const name = path.basename(file, path.extname(file));
        const extension = path.extname(file).substring(1);
        
        return {
          name,
          displayName: formatComponentName(name),
          path: `/components/${name}`,
          extension
        };
      });
  } catch (error) {
    console.error("Error reading components directory:", error);
    return [];
  }
}

export default function Home() {
  const components = getComponentFiles();
  
  return (
    <div className="min-h-screen p-8">
      <header className="mb-12">
        <h1 className="text-3xl font-bold mb-2">Component Viewer</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Browse and view components from the external components directory
        </p>
      </header>
      
      <main>
        {components.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {components.map((component) => (
              <Link 
                href={component.path} 
                key={component.name}
                className="block p-6 border border-gray-200 dark:border-gray-800 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-900 transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-xl font-semibold">{component.displayName}</h2>
                  <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded">
                    {component.extension}
                  </span>
                </div>
                <p className="text-gray-600 dark:text-gray-400">
                  Click to view this component
                </p>
              </Link>
            ))}
          </div>
        ) : (
          <div className="text-center p-12 border border-dashed border-gray-300 dark:border-gray-700 rounded-lg">
            <h2 className="text-xl font-medium mb-2">No Components Found</h2>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              No TSX or HTML files were found in the components directory.
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-500">
              Make sure the components directory exists at:
              <br />
              <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded mt-2 block">
                {path.join(process.cwd(), "..", "components")}
              </code>
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
