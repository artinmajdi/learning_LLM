import React from 'react';

const AdapterExplained = () => {
  return (
    <div className="p-6 bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 rounded-xl shadow-lg">
      <h2 className="text-2xl font-bold text-center mb-6 bg-white bg-opacity-70 p-3 rounded-lg shadow-sm text-indigo-800">How Adapters Really Work: Common Misconceptions Cleared</h2>
      
      <div className="grid grid-cols-1 gap-8">
        {/* Misconception 1 */}
        <div className="bg-white rounded-xl overflow-hidden shadow-md">
          <div className="bg-red-100 p-4 border-l-4 border-red-500">
            <h3 className="text-lg font-bold text-red-700">Misconception #1: Using PCA for Dimensionality Reduction</h3>
            <p className="text-red-600 italic mt-1">❌ Not using dimension reduction techniques like PCA</p>
          </div>
          
          <div className="p-5">
            <div className="bg-green-50 p-4 rounded-lg mb-4 border-l-4 border-green-500">
              <h3 className="text-lg font-bold text-green-700">Actual Mechanism: Learned Linear Projection</h3>
              <p className="text-green-600 mt-1">✅ The bottleneck is created using a fully learnable weight matrix</p>
            </div>
            
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1 bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold text-gray-700 mb-2">Down-Projection Matrix</h4>
                <p className="text-sm text-gray-600">
                  A fully trainable weight matrix W_down of shape [768×64] that transforms input features.
                </p>
                <div className="mt-3 font-mono text-xs bg-gray-100 p-2 rounded">
                  y = W_down · x + b_down
                </div>
                <p className="mt-2 text-xs text-gray-500">
                  Unlike PCA which finds principal components, this matrix is randomly initialized and then learned through gradient descent to minimize the task-specific loss function.
                </p>
              </div>
              
              <div className="flex-1 bg-gray-50 p-4 rounded-lg">
                <h4 className="font-semibold text-gray-700 mb-2">What is Being Learned?</h4>
                <ul className="text-sm text-gray-600 list-disc pl-4">
                  <li>Which input features are most relevant for the specific task</li>
                  <li>How to compress 768D information into task-essential 64D representation</li>
                  <li>The optimal feature combinations for the downstream task</li>
                </ul>
                <p className="mt-2 text-xs text-gray-500 italic">
                  This tiny bottleneck forces the model to learn only task-relevant transformations since it cannot pass all information through.
                </p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Misconception 2 */}
        <div className="bg-white rounded-xl overflow-hidden shadow-md">
          <div className="bg-red-100 p-4 border-l-4 border-red-500">
            <h3 className="text-lg font-bold text-red-700">Misconception #2: Training the 64 Features Directly</h3>
            <p className="text-red-600 italic mt-1">❌ We do not directly train the bottleneck features themselves</p>
          </div>
          
          <div className="p-5">
            <div className="bg-green-50 p-4 rounded-lg mb-4 border-l-4 border-green-500">
              <h3 className="text-lg font-bold text-green-700">Actual Mechanism: Training the Projection Matrices</h3>
              <p className="text-green-600 mt-1">✅ We train the weight matrices that create and process the bottleneck</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-700 mb-2">1. Forward Pass</h4>
                <p className="text-sm text-blue-600">
                  Input → Down-project → Activate → Up-project → Add to original
                </p>
                <p className="mt-2 text-xs text-blue-500">
                  During inference, features flow through the adapter, but we never directly manipulate the 64D bottleneck values.
                </p>
              </div>
              
              <div className="bg-purple-50 p-4 rounded-lg">
                <h4 className="font-semibold text-purple-700 mb-2">2. Loss Calculation</h4>
                <p className="text-sm text-purple-600">
                  Model output compared to target (e.g., next token prediction)
                </p>
                <p className="mt-2 text-xs text-purple-500">
                  Loss tells us how to adjust adapters to better solve the task at hand.
                </p>
              </div>
              
              <div className="bg-pink-50 p-4 rounded-lg">
                <h4 className="font-semibold text-pink-700 mb-2">3. Backpropagation</h4>
                <p className="text-sm text-pink-600">
                  Gradients flow backward, updating W_down and W_up matrices
                </p>
                <p className="mt-2 text-xs text-pink-500">
                  Base model parameters remain frozen! Only adapter matrices receive updates.
                </p>
              </div>
            </div>
            
            <div className="mt-4 bg-yellow-50 p-3 rounded-lg border border-yellow-200">
              <p className="text-sm text-yellow-700">
                <span className="font-semibold">Bottom line:</span> We are not directly optimizing the 64 bottleneck features - we are optimizing the matrices that transform features into and out of the bottleneck. The 64D space is just an intermediate representation.
              </p>
            </div>
          </div>
        </div>
        
        {/* Misconception 3 */}
        <div className="bg-white rounded-xl overflow-hidden shadow-md">
          <div className="bg-red-100 p-4 border-l-4 border-red-500">
            <h3 className="text-lg font-bold text-red-700">Misconception #3: Replacing Input Features</h3>
            <p className="text-red-600 italic mt-1">❌ Adapters do not replace the original features</p>
          </div>
          
          <div className="p-5">
            <div className="bg-green-50 p-4 rounded-lg mb-4 border-l-4 border-green-500">
              <h3 className="text-lg font-bold text-green-700">Actual Mechanism: Residual Addition within the Layer</h3>
              <p className="text-green-600 mt-1">✅ Adapters create a parallel path that gets added back to the main data flow</p>
            </div>
            
            <div className="flex flex-col space-y-4">
              <div className="bg-indigo-50 p-4 rounded-lg">
                <h4 className="font-semibold text-indigo-700 mb-2">Residual Connection, Not Input Replacement</h4>
                <div className="flex flex-col md:flex-row items-center gap-4">
                  <div className="text-sm text-indigo-600 flex-1">
                    <p className="mb-2">The adapter does not replace the input features. Instead:</p>
                    <ol className="list-decimal pl-5">
                      <li>Original features flow through the transformer layer</li>
                      <li>In parallel, those same features also flow through the adapter</li>
                      <li>The adapter output is then added to the main pathway (not replacing it)</li>
                      <li>This happens inside the transformer layer, not between layers</li>
                    </ol>
                  </div>
                </div>
              </div>
              
              <div className="bg-teal-50 p-4 rounded-lg">
                <h4 className="font-semibold text-teal-700 mb-2">Why This Works</h4>
                <p className="text-sm text-teal-600 mb-2">
                  This residual connection design is brilliant because:
                </p>
                <ul className="text-sm text-teal-600 list-disc pl-4">
                  <li>It preserves all the original model capabilities since the main path is unchanged</li>
                  <li>The adapter only needs to learn the adjustment needed for the specific task</li>
                  <li>If the adapter output is zero, the model works identically to the original</li>
                  <li>This design allows for stable training since the model starts from a good initialization</li>
                </ul>
              </div>
              
              <div className="bg-amber-50 p-4 rounded-lg">
                <h4 className="font-semibold text-amber-700 mb-2">Scaling Factor</h4>
                <p className="text-sm text-amber-600">
                  Often adapters use a small scaling factor (α ≈ 0.1) when adding adapter outputs back to the main path. This prevents the adapter from dominating the forward pass early in training, helping stability.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-8 bg-white p-5 rounded-lg shadow-inner">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Key Takeaways for American Airlines AI Systems</h3>
        <div className="text-gray-700">
          <p>
            Adapters provide an efficient way to customize large models for specific airline operations without the massive computational cost of full fine-tuning. Their innovative design creates a perfect balance between:
          </p>
          <ul className="list-disc pl-5 mt-2">
            <li><span className="font-semibold text-indigo-700">Parameter Efficiency:</span> 99.9% of model parameters remain frozen</li>
            <li><span className="font-semibold text-purple-700">Knowledge Preservation:</span> Base model capabilities remain intact due to residual connections</li>
            <li><span className="font-semibold text-pink-700">Specialization:</span> Each adapter learns a specific task (e.g., maintenance documentation, flight operations)</li>
            <li><span className="font-semibold text-blue-700">Deployment Flexibility:</span> Swap adapters based on department or use case, without changing the base model</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AdapterExplained;
