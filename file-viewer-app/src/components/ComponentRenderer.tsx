'use client';

import React, { useEffect, useState } from 'react';

interface ComponentRendererProps {
  content: string;
  type: 'tsx' | 'html';
  name: string;
}

const ComponentRenderer: React.FC<ComponentRendererProps> = ({ content, type, name }) => {
  const [renderedContent, setRenderedContent] = useState<React.ReactNode>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setIsLoading(true);
    setError(null);

    try {
      if (type === 'tsx') {
        // Display TSX as code (e.g., inside a <pre> tag or using a syntax highlighter)
        setRenderedContent(
          <div className="p-4 bg-gray-900 text-white rounded-md overflow-auto text-sm font-mono">
            <pre><code>{content}</code></pre>
          </div>
        );
      } else if (type === 'html') {
        // Use iframe to render HTML content, allowing its scripts to run in a sandboxed environment
        setRenderedContent(
          <iframe
            srcDoc={content}
            title={name || 'HTML Component Preview'}
            style={{ width: '100%', height: '600px', border: '1px solid #ccc', borderRadius: '4px' }}
            sandbox="allow-scripts allow-same-origin allow-popups allow-forms" // Adjust sandbox as needed
          />
        );
      } else {
        setError(`Unsupported component type: ${type}`);
        setRenderedContent(null);
      }
    } catch (e: any) {
      console.error(`Error processing component ${name} (type: ${type}):`, e);
      setError(`Failed to process component: ${e.message}. Check console for details.`);
      // Optionally, display the raw content if processing fails before error/loading states handle it
      setRenderedContent(
        <div className="mt-4 p-3 bg-red-100 dark:bg-red-900 border border-red-400 dark:border-red-700 rounded text-xs overflow-auto max-h-[300px]">
          <p className="font-semibold text-red-700 dark:text-red-300">Raw content due to processing error:</p>
          <pre><code>{content}</code></pre>
        </div>
      );
    } finally {
      setIsLoading(false);
    }
  }, [content, type, name]);

  if (error) {
    return (
      <div className="p-4 border border-red-300 bg-red-50 text-red-700 rounded-md dark:bg-red-900 dark:text-red-200 dark:border-red-700">
        <h3 className="font-semibold mb-2">Error Rendering Component: {name}</h3>
        <p className="text-sm mb-2">{error}</p>
        {/* Displaying the content that caused the error can be helpful for debugging */}
        <details className="mt-2 text-xs">
          <summary className="cursor-pointer">Show Raw Content</summary>
          <div className="mt-1 p-2 bg-gray-100 dark:bg-gray-800 rounded overflow-auto max-h-[200px]">
            <pre><code>{content}</code></pre>
          </div>
        </details>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8 min-h-[300px]">
        <div className="animate-pulse text-gray-500 dark:text-gray-400">
          Loading component preview...
        </div>
      </div>
    );
  }

  return (
    <div className="component-renderer w-full h-full min-h-[600px]">
      {renderedContent}
    </div>
  );
};

export default ComponentRenderer;
