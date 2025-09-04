import React from 'react';

interface SlotCardProps {
  id: number;
  salience: number;
  age: number;
  content: string;
  level?: number;
  access_count?: number;
  compression_ratio?: number;
}

const SlotCard: React.FC<SlotCardProps> = ({ 
  id, 
  salience, 
  age, 
  content, 
  level = 0, 
  access_count = 0,
  compression_ratio = 1 
}) => {
  const salienceColor = `rgba(34, 197, 94, ${salience})`; // Green with opacity
  const levelColors = ['#3b82f6', '#8b5cf6', '#f59e0b']; // Blue, Purple, Orange
  const levelColor = levelColors[level] || '#6b7280';

  const getAgeColor = (age: number) => {
    if (age < 20) return 'text-green-400';
    if (age < 50) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="bg-gray-700 p-3 rounded-md shadow-lg border-l-4 hover:bg-gray-600 transition-colors cursor-pointer group" 
         style={{ borderColor: salienceColor }}>
      
      {/* Header with ID and level */}
      <div className="flex justify-between items-center mb-2">
        <span className="font-bold text-lg">#{id}</span>
        <div className="flex items-center space-x-2">
          <span 
            className="text-xs px-2 py-1 rounded-full font-semibold"
            style={{ backgroundColor: levelColor, color: 'white' }}
          >
            L{level}
          </span>
          {compression_ratio > 1 && (
            <span className="text-xs text-orange-400">
              {compression_ratio}x
            </span>
          )}
        </div>
      </div>

      {/* Content preview */}
      <p className="text-xs text-gray-300 truncate mb-3 group-hover:text-white transition-colors">
        {content}
      </p>

      {/* Metrics */}
      <div className="space-y-2">
        {/* Salience bar */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-400">Salience</span>
            <span className="text-green-400 font-mono">{salience.toFixed(3)}</span>
          </div>
          <div className="w-full bg-gray-600 rounded-full h-2">
            <div 
              className="bg-green-500 h-2 rounded-full transition-all duration-300" 
              style={{ width: `${Math.min(100, salience * 100)}%` }}
            ></div>
          </div>
        </div>

        {/* Age and Access Count */}
        <div className="flex justify-between items-center text-xs">
          <div className="flex items-center space-x-1">
            <span className="text-gray-400">Age:</span>
            <span className={`font-mono ${getAgeColor(age)}`}>{age}</span>
          </div>
          <div className="flex items-center space-x-1">
            <span className="text-gray-400">Access:</span>
            <span className="text-cyan-400 font-mono">{access_count}</span>
          </div>
        </div>

        {/* Activity indicator */}
        <div className="flex justify-between items-center text-xs">
          <span className="text-gray-400">Activity</span>
          <div className="flex space-x-1">
            {[...Array(5)].map((_, i) => (
              <div
                key={i}
                className={`w-1 h-3 rounded-full ${
                  i < Math.floor(salience * 5) ? 'bg-green-500' : 'bg-gray-600'
                }`}
              ></div>
            ))}
          </div>
        </div>
      </div>

      {/* Hover overlay with detailed info */}
      <div className="invisible group-hover:visible absolute z-20 bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-xl transform -translate-y-2 left-0 right-0 mx-2">
        <div className="text-sm space-y-1">
          <div className="font-semibold text-cyan-400">Slot #{id} Details</div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-gray-400">Level:</span>
              <span className="text-white ml-1">{level}</span>
            </div>
            <div>
              <span className="text-gray-400">Compression:</span>
              <span className="text-orange-400 ml-1">{compression_ratio}x</span>
            </div>
            <div>
              <span className="text-gray-400">Salience:</span>
              <span className="text-green-400 ml-1">{salience.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-gray-400">Age:</span>
              <span className={`ml-1 ${getAgeColor(age)}`}>{age}</span>
            </div>
            <div className="col-span-2">
              <span className="text-gray-400">Access Count:</span>
              <span className="text-cyan-400 ml-1">{access_count}</span>
            </div>
          </div>
          <div className="mt-2">
            <span className="text-gray-400">Content:</span>
            <div className="text-white text-xs mt-1 bg-gray-900 p-2 rounded">
              {content}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SlotCard;
