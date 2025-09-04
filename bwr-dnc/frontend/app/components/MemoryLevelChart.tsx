import React from 'react';

interface Slot {
  id: number;
  level: number;
  salience: number;
  age: number;
  access_count: number;
  compression_ratio: number;
}

interface MemoryLevelChartProps {
  slots: Slot[];
}

const MemoryLevelChart: React.FC<MemoryLevelChartProps> = ({ slots }) => {
  const levels = [0, 1, 2];
  
  const getLevelStats = (level: number) => {
    const levelSlots = slots.filter(s => s.level === level);
    return {
      count: levelSlots.length,
      avgSalience: levelSlots.reduce((acc, s) => acc + s.salience, 0) / levelSlots.length || 0,
      avgAge: levelSlots.reduce((acc, s) => acc + s.age, 0) / levelSlots.length || 0,
      totalAccess: levelSlots.reduce((acc, s) => acc + s.access_count, 0),
      activeSlots: levelSlots.filter(s => s.salience > 0.1).length
    };
  };

  return (
    <div className="bg-gray-800 p-6 rounded-lg">
      <h3 className="text-xl font-semibold mb-4 text-cyan-400">Memory Level Distribution</h3>
      
      <div className="grid grid-cols-3 gap-6">
        {levels.map(level => {
          const stats = getLevelStats(level);
          const compressionRatio = [1, 2, 4][level];
          
          return (
            <div key={level} className="bg-gray-900 p-4 rounded-lg">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-lg font-semibold text-cyan-300">
                  Level {level}
                </h4>
                <span className="text-sm text-orange-400">
                  {compressionRatio}x compression
                </span>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Slots:</span>
                  <span className="text-white font-mono">{stats.count}</span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Active:</span>
                  <span className="text-green-400 font-mono">{stats.activeSlots}</span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Avg Salience:</span>
                  <span className="text-purple-400 font-mono">
                    {stats.avgSalience.toFixed(3)}
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Avg Age:</span>
                  <span className="text-blue-400 font-mono">
                    {stats.avgAge.toFixed(1)}
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Total Access:</span>
                  <span className="text-yellow-400 font-mono">{stats.totalAccess}</span>
                </div>
                
                {/* Visual progress bar for salience */}
                <div className="mt-3">
                  <div className="text-xs text-gray-500 mb-1">Salience Distribution</div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${Math.min(100, stats.avgSalience * 100)}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Overall memory usage visualization */}
      <div className="mt-6 bg-gray-900 p-4 rounded-lg">
        <h4 className="text-lg font-semibold mb-3 text-cyan-300">Memory Usage Overview</h4>
        <div className="grid grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {slots.filter(s => s.salience > 0.1).length}
            </div>
            <div className="text-sm text-gray-400">Active Slots</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">
              {(slots.reduce((acc, s) => acc + s.salience, 0) / slots.length).toFixed(2)}
            </div>
            <div className="text-sm text-gray-400">Avg Salience</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-400">
              {slots.reduce((acc, s) => acc + s.access_count, 0)}
            </div>
            <div className="text-sm text-gray-400">Total Accesses</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-400">
              {((slots.reduce((acc, s) => acc + s.compression_ratio, 0) / slots.length) || 1).toFixed(1)}x
            </div>
            <div className="text-sm text-gray-400">Avg Compression</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MemoryLevelChart;
