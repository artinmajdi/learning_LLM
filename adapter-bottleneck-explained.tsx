import React from 'react';

const AdapterBottleneckExplained = () => {
  return (
    <div className="p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl shadow-lg">
      <h2 className="text-xl font-bold text-center mb-8 text-indigo-800">The Adapter Bottleneck: A Visual Explanation</h2>
      
      <div className="grid grid-cols-1 gap-8">
        {/* Input Features Visualization */}
        <div className="grid grid-cols-12 gap-2 items-center">
          <div className="col-span-2">
            <div className="text-right font-semibold text-gray-700">Input Features<br/>(768 dimensions)</div>
          </div>
          <div className="col-span-10 flex justify-start">
            <div className="flex space-x-1">
              {Array(40).fill(0).map((_, i) => (
                <div key={i} className="w-4 h-16 bg-blue-400 rounded-sm opacity-80"></div>
              ))}
              <div className="w-8 flex items-center justify-center text-gray-500">...</div>
            </div>
          </div>
        </div>
        
        {/* Down-Project Step */}
        <div className="grid grid-cols-12 gap-2 items-center">
          <div className="col-span-2">
            <div className="bg-green-100 p-3 rounded-lg border-2 border-green-400 shadow-md text-center">
              <div className="mx-auto mb-2 text-green-600 flex justify-center">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="12" cy="12" r="10" stroke="#22c55e" strokeWidth="2" fill="#dcfce7" />
                  <path d="M8 12L12 16L16 12" stroke="#16a34a" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M12 8L12 16" stroke="#16a34a" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <div className="font-bold text-green-800">DOWN-PROJECT</div>
              <div className="text-sm text-green-700 mt-1">Linear Layer</div>
            </div>
          </div>
          <div className="col-span-1 flex justify-center">
            <div className="w-6 h-16 bg-gray-300 rounded-md flex items-center justify-center">
              <svg height="30" width="20" className="transform rotate-90">
                <path d="M1 5 L10 15 L1 25" stroke="gray" strokeWidth="2" fill="none"/>
              </svg>
            </div>
          </div>
          <div className="col-span-9">
            <div className="bg-green-50 p-3 rounded-lg">
              <p className="text-green-800 mb-2">
                <span className="font-semibold">Function:</span> Compresses the high-dimensional input into a much smaller representation
              </p>
              <p className="text-green-700 text-sm">
                • Takes 768-dimension vectors and projects them to just 64 dimensions (or even fewer)
                <br/>
                • This is a simple linear transformation: y = Wx + b
                <br/>
                • By creating this "information bottleneck," we force the model to learn only essential patterns
              </p>
            </div>
          </div>
        </div>
        
        {/* Bottleneck Visualization */}
        <div className="grid grid-cols-12 gap-2 items-center">
          <div className="col-span-2">
            <div className="text-right font-semibold text-gray-700">Bottleneck<br/>(64 dimensions)</div>
          </div>
          <div className="col-span-10 flex justify-start">
            <div className="flex space-x-1">
              {Array(12).fill(0).map((_, i) => (
                <div key={i} className="w-4 h-8 bg-green-500 rounded-sm opacity-80"></div>
              ))}
              <div className="w-8 flex items-center justify-center text-gray-500">...</div>
            </div>
          </div>
        </div>
        
        {/* Activate Step */}
        <div className="grid grid-cols-12 gap-2 items-center">
          <div className="col-span-2">
            <div className="bg-yellow-100 p-3 rounded-lg border-2 border-yellow-400 shadow-md text-center">
              <div className="mx-auto mb-2 text-yellow-600 flex justify-center">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" stroke="#ca8a04" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="#fef08a"/>
                </svg>
              </div>
              <div className="font-bold text-yellow-800">ACTIVATE</div>
              <div className="text-sm text-yellow-700 mt-1">Non-linearity</div>
            </div>
          </div>
          <div className="col-span-1 flex justify-center">
            <div className="w-6 h-16 bg-gray-300 rounded-md flex items-center justify-center">
              <svg height="30" width="20" className="transform rotate-90">
                <path d="M1 5 L10 15 L1 25" stroke="gray" strokeWidth="2" fill="none"/>
              </svg>
            </div>
          </div>
          <div className="col-span-9">
            <div className="bg-yellow-50 p-3 rounded-lg">
              <p className="text-yellow-800 mb-2">
                <span className="font-semibold">Function:</span> Adds non-linearity to enable complex pattern modeling
              </p>
              <p className="text-yellow-700 text-sm">
                • Applies a non-linear function (usually GELU or ReLU) to each value in the bottleneck
                <br/>
                • Without this step, the entire adapter would just be a linear transformation
                <br/>
                • The non-linearity is what gives the adapter its expressive power to learn complex patterns
              </p>
            </div>
          </div>
        </div>
        
        {/* Activated Bottleneck Visualization */}
        <div className="grid grid-cols-12 gap-2 items-center">
          <div className="col-span-2">
            <div className="text-right font-semibold text-gray-700">Activated<br/>Features</div>
          </div>
          <div className="col-span-10 flex justify-start">
            <div className="flex space-x-1">
              {Array(12).fill(0).map((_, i) => (
                <div key={i} 
                  className="w-4 bg-yellow-500 rounded-sm opacity-80" 
                  style={{
                    height: Math.floor(Math.random() * 8) + 3 + 'px', 
                    marginTop: 8 - (Math.floor(Math.random() * 8) + 3) + 'px'
                  }}></div>
              ))}
              <div className="w-8 flex items-center justify-center text-gray-500">...</div>
            </div>
          </div>
        </div>
        
        {/* Up-Project Step */}
        <div className="grid grid-cols-12 gap-2 items-center">
          <div className="col-span-2">
            <div className="bg-purple-100 p-3 rounded-lg border-2 border-purple-400 shadow-md text-center">
              <div className="mx-auto mb-2 text-purple-600 flex justify-center">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="12" cy="12" r="10" stroke="#9333ea" strokeWidth="2" fill="#f3e8ff" />
                  <path d="M8 12L12 8L16 12" stroke="#7e22ce" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M12 8L12 16" stroke="#7e22ce" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <div className="font-bold text-purple-800">UP-PROJECT</div>
              <div className="text-sm text-purple-700 mt-1">Linear Layer</div>
            </div>
          </div>
          <div className="col-span-1 flex justify-center">
            <div className="w-6 h-16 bg-gray-300 rounded-md flex items-center justify-center">
              <svg height="30" width="20" className="transform rotate-90">
                <path d="M1 5 L10 15 L1 25" stroke="gray" strokeWidth="2" fill="none"/>
              </svg>
            </div>
          </div>
          <div className="col-span-9">
            <div className="bg-purple-50 p-3 rounded-lg">
              <p className="text-purple-800 mb-2">
                <span className="font-semibold">Function:</span> Expands the bottleneck representation back to original dimensionality
              </p>
              <p className="text-purple-700 text-sm">
                • Projects the 64-dimension bottleneck back to 768 dimensions to match the original network
                <br/>
                • Another linear transformation: z = Vy + c
                <br/>
                • Creates a modified version of the original features, focused on the task-specific patterns
              </p>
            </div>
          </div>
        </div>
        
        {/* Output Features Visualization */}
        <div className="grid grid-cols-12 gap-2 items-center">
          <div className="col-span-2">
            <div className="text-right font-semibold text-gray-700">Output Features<br/>(768 dimensions)</div>
          </div>
          <div className="col-span-10 flex justify-start">
            <div className="flex space-x-1">
              {Array(40).fill(0).map((_, i) => (
                <div key={i} className="w-4 h-16 bg-purple-400 rounded-sm opacity-80"></div>
              ))}
              <div className="w-8 flex items-center justify-center text-gray-500">...</div>
            </div>
          </div>
        </div>
        
        {/* Residual Connection */}
        <div className="grid grid-cols-12 gap-2 items-center">
          <div className="col-span-2">
            <div className="bg-indigo-100 p-3 rounded-lg border-2 border-indigo-400 shadow-md text-center">
              <div className="font-bold text-indigo-800">RESIDUAL ADD</div>
              <div className="text-sm text-indigo-700 mt-1">Original + Adapter</div>
            </div>
          </div>
          <div className="col-span-10">
            <div className="bg-indigo-50 p-3 rounded-lg">
              <p className="text-indigo-800 mb-2">
                <span className="font-semibold">Final Step:</span> The adapter output is <span className="font-bold">added</span> to the original transformer features (not replacing them)
              </p>
              <p className="text-indigo-700 text-sm">
                This residual connection ensures that the adapter only needs to learn the "difference" or "adjustment" needed for the specific task, rather than recreating all the knowledge in the original model.
              </p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-8 bg-white p-4 rounded-lg shadow-inner">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Key Takeaway: The Bottleneck Principle</h3>
        <p className="text-gray-700">
          By forcing information through a narrow bottleneck (from 768→64→768 dimensions), adapters make the model learn efficient, task-specific representations with very few parameters. This is much more efficient than fine-tuning the entire model while still allowing for powerful task adaptation.
        </p>
      </div>
    </div>
  );
};

export default AdapterBottleneckExplained;
