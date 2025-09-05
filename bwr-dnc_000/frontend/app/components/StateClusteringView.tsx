import React, { useState, useEffect } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ScatterController } from 'chart.js';
import { Scatter } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ScatterController);

interface StateCluster {
  id: number;
  centroid: number[];
  members: number[];
  size: number;
  variance: number;
}

interface ClusteringData {
  clusters: StateCluster[];
  total_states: number;
  timestamp: string;
}

const StateClusteringView: React.FC = () => {
  const [clusteringData, setClusteringData] = useState<ClusteringData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchClusteringData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/research/state-clustering');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setClusteringData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      console.error('Failed to fetch clustering data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchClusteringData();
  }, []);

  const generateScatterData = () => {
    if (!clusteringData?.clusters) return { datasets: [] };

    const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];
    
    const datasets = clusteringData.clusters.map((cluster, index) => {
      // Project high-dimensional centroids to 2D for visualization
      // Use first two dimensions or create a simple projection
      const x = cluster.centroid[0] || Math.random() * 100;
      const y = cluster.centroid[1] || Math.random() * 100;
      
      return {
        label: `Cluster ${cluster.id} (${cluster.size} states)`,
        data: [{
          x: x,
          y: y,
          size: cluster.size,
          variance: cluster.variance
        }],
        backgroundColor: colors[index % colors.length],
        borderColor: colors[index % colors.length],
        pointRadius: Math.max(5, cluster.size * 2),
        showLine: false
      };
    });

    return { datasets };
  };

  const scatterOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'State Space Clustering (2D Projection)'
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            const point = context.raw;
            return [
              `Cluster Size: ${point.size}`,
              `Variance: ${point.variance.toFixed(4)}`,
              `Position: (${point.x.toFixed(3)}, ${point.y.toFixed(3)})`
            ];
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Principal Component 1'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Principal Component 2'
        }
      }
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">State Clustering Analysis</h2>
        <button
          onClick={fetchClusteringData}
          disabled={loading}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Analyzing...' : 'Refresh Clustering'}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          <strong>Error:</strong> {error}
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-gray-600">Performing state clustering analysis...</span>
        </div>
      )}

      {clusteringData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Clustering Visualization */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div style={{ height: '400px' }}>
              <Scatter data={generateScatterData()} options={scatterOptions} />
            </div>
          </div>

          {/* Cluster Statistics */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold mb-4">Cluster Statistics</h3>
            <div className="space-y-4">
              <div className="text-sm text-gray-600">
                <strong>Total States Analyzed:</strong> {clusteringData.total_states}
              </div>
              <div className="text-sm text-gray-600">
                <strong>Number of Clusters:</strong> {clusteringData.clusters.length}
              </div>
              <div className="text-sm text-gray-600">
                <strong>Last Updated:</strong> {new Date(clusteringData.timestamp).toLocaleString()}
              </div>
            </div>

            {/* Individual Cluster Details */}
            <div className="mt-6">
              <h4 className="font-medium mb-3">Cluster Details</h4>
              <div className="space-y-3">
                {clusteringData.clusters.map((cluster, index) => (
                  <div key={cluster.id} className="border rounded p-3 bg-gray-50">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-medium">Cluster {cluster.id}</div>
                        <div className="text-sm text-gray-600">
                          {cluster.size} states
                        </div>
                      </div>
                      <div className="text-right text-sm">
                        <div>Variance: {cluster.variance.toFixed(4)}</div>
                        <div>Members: {cluster.members.join(', ')}</div>
                      </div>
                    </div>
                    <div className="mt-2 text-xs text-gray-500">
                      Centroid dimensions: {cluster.centroid.length}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Analysis Insights */}
      {clusteringData && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold mb-4">Clustering Insights</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 p-4 rounded">
              <div className="text-2xl font-bold text-blue-600">
                {clusteringData.clusters.length}
              </div>
              <div className="text-sm text-blue-700">Distinct State Regions</div>
            </div>
            <div className="bg-green-50 p-4 rounded">
              <div className="text-2xl font-bold text-green-600">
                {Math.max(...clusteringData.clusters.map(c => c.size))}
              </div>
              <div className="text-sm text-green-700">Largest Cluster Size</div>
            </div>
            <div className="bg-purple-50 p-4 rounded">
              <div className="text-2xl font-bold text-purple-600">
                {(clusteringData.clusters.reduce((sum, c) => sum + c.variance, 0) / clusteringData.clusters.length).toFixed(3)}
              </div>
              <div className="text-sm text-purple-700">Average Variance</div>
            </div>
          </div>
          
          <div className="mt-4 text-sm text-gray-600">
            <p><strong>Interpretation:</strong> This clustering analysis reveals how the model's internal states organize during processing. 
            Tighter clusters (lower variance) indicate more consistent state patterns, while scattered clusters suggest diverse processing modes.</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default StateClusteringView;
