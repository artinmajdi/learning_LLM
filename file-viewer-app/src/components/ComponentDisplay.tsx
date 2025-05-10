'use client';

import React from 'react';
import ComponentRenderer from './ComponentRenderer';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { coy } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface ComponentDisplayProps {
  content: string;
  type: 'tsx' | 'html';
  name: string;
}

const ComponentDisplay: React.FC<ComponentDisplayProps> = ({ content, type, name }) => {
  if (type === 'tsx') {
    return (
      <div className="p-1 bg-gray-50 dark:bg-gray-800 rounded-md shadow-inner text-sm max-h-[70vh] overflow-auto">
        <SyntaxHighlighter 
          language="tsx" 
          style={coy} 
          showLineNumbers 
          wrapLines
          customStyle={{ margin: 0, padding: '1rem', borderRadius: '0.375rem' }}
        >
          {content.trim()}
        </SyntaxHighlighter>
      </div>
    );
  }

  if (type === 'html') {
    return <ComponentRenderer content={content} type={type} name={name} />;
  }

  return (
    <div className="p-4 text-red-500">
      Unsupported component type: {type}
    </div>
  );
};

export default ComponentDisplay;
