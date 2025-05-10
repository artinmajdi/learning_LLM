import React from 'react';

const ComparisonTable = () => {
  return (
    <div className="flex flex-col w-full">
      <div className="overflow-x-auto">
        <div className="bg-gradient-to-r from-orange-50 to-blue-50 p-6 rounded-lg shadow-lg">
          <h2 className="text-xl font-bold text-center mb-6 bg-white bg-opacity-70 p-3 rounded-md shadow-sm">LoRA vs. Adapters: Technical Comparison</h2>
          
          <table className="min-w-full bg-white rounded-lg overflow-hidden">
            <thead className="bg-gradient-to-r from-orange-200 to-orange-300 text-gray-800">
              <tr>
                <th className="py-3 px-4 text-left font-semibold border-b border-orange-400">Aspect</th>
                <th className="py-3 px-4 text-left font-semibold border-b border-orange-400">LoRA (Approach 1)</th>
                <th className="py-3 px-4 text-left font-semibold border-b border-blue-400">Adapters (Approach 2)</th>
              </tr>
            </thead>
            <tbody>
              <tr className="hover:bg-gray-50 transition-colors duration-200">
                <td className="py-3 px-4 border-b border-gray-200 font-medium">Core Concept</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-orange-50">Matrix factorization trick (A×B)</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-blue-50">New bottleneck layers</td>
              </tr>
              <tr className="hover:bg-gray-50 transition-colors duration-200">
                <td className="py-3 px-4 border-b border-gray-200 font-medium">Model Architecture</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-orange-50">Unchanged (same structure)</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-blue-50">Modified (adds new layers)</td>
              </tr>
              <tr className="hover:bg-gray-50 transition-colors duration-200">
                <td className="py-3 px-4 border-b border-gray-200 font-medium">Parameter Location</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-orange-50">Parallel to original weights</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-blue-50">Sequential in network flow</td>
              </tr>
              <tr className="hover:bg-gray-50 transition-colors duration-200">
                <td className="py-3 px-4 border-b border-gray-200 font-medium">Computation Pattern</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-orange-50">W + A×B (addition)</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-blue-50">Down-project → Activate → Up-project</td>
              </tr>
              <tr className="hover:bg-gray-50 transition-colors duration-200">
                <td className="py-3 px-4 border-b border-gray-200 font-medium">Implementation Complexity</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-orange-50">Lower (simple matrix math)</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-blue-50">Higher (architectural changes)</td>
              </tr>
              <tr className="hover:bg-gray-50 transition-colors duration-200">
                <td className="py-3 px-4 border-b border-gray-200 font-medium">Typical Parameter Count</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-orange-50">~0.1-1% of base model</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-blue-50">~1-3% of base model</td>
              </tr>
              <tr className="hover:bg-gray-50 transition-colors duration-200">
                <td className="py-3 px-4 border-b border-gray-200 font-medium">Modularity</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-orange-50">Moderate (weights are specific)</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-blue-50">High (modules can be swapped)</td>
              </tr>
              <tr className="hover:bg-gray-50 transition-colors duration-200">
                <td className="py-3 px-4 border-b border-gray-200 font-medium">Popular Implementations</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-orange-50">HuggingFace PEFT, Microsoft LoRA</td>
                <td className="py-3 px-4 border-b border-gray-200 bg-blue-50">AdapterHub, HuggingFace Adapters</td>
              </tr>
              <tr className="hover:bg-gray-50 transition-colors duration-200">
                <td className="py-3 px-4 border-gray-200 font-medium">Key Advantage</td>
                <td className="py-3 px-4 border-gray-200 bg-orange-50">Extremely memory efficient</td>
                <td className="py-3 px-4 border-gray-200 bg-blue-50">Highly modular, easy to swap tasks</td>
              </tr>
            </tbody>
          </table>
          
          <div className="mt-6 bg-white bg-opacity-70 p-4 rounded-md text-sm text-gray-700">
            <strong className="text-gray-900">Note:</strong> Both approaches achieve similar goals (parameter-efficient fine-tuning) but through fundamentally different mechanisms. LoRA is generally more popular for LLM fine-tuning due to its implementation simplicity and lower parameter count.
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComparisonTable;
