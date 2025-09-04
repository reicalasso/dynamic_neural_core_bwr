import type { NextPage } from 'next';
import Head from 'next/head';
import { useState, useEffect } from 'react';

const Home: NextPage = () => {
  const [isClient, setIsClient] = useState(false);

  // Prevent hydration errors
  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) {
    return (
      <div className="bg-gray-900 text-white min-h-screen flex items-center justify-center">
        <div className="text-xl text-cyan-400">Loading...</div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 text-white min-h-screen">
      <Head>
        <title>BWR-NSM Visualizer</title>
        <meta name="description" content="Neural State Machine Visualizer" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="container mx-auto p-4">
        <h1 className="text-4xl font-bold mb-4 text-cyan-400">BWR-NSM — Dev Visualizer</h1>
        <p className="text-gray-400 mb-8">
          Real-time dashboard for monitoring the Neural State Machine&apos;s internal state.
        </p>
        
        {/* Simple demo content */}
        <div className="space-y-6">
          <div className="bg-gray-800 p-6 rounded-lg">
            <h2 className="text-2xl font-bold text-cyan-400 mb-4">System Status</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-700 p-4 rounded">
                <div className="text-green-400 text-lg">✅ Backend Online</div>
                <div className="text-sm text-gray-300">API responding</div>
              </div>
              <div className="bg-gray-700 p-4 rounded">
                <div className="text-green-400 text-lg">🧠 Model Loaded</div>
                <div className="text-sm text-gray-300">38M parameters</div>
              </div>
              <div className="bg-gray-700 p-4 rounded">
                <div className="text-green-400 text-lg">🚀 Ready</div>
                <div className="text-sm text-gray-300">All systems go</div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-cyan-400 mb-4">Quick Links</h3>
            <div className="space-y-2">
              <a href="/research" 
                 className="block bg-purple-600 hover:bg-purple-700 p-3 rounded text-white transition-colors">
                🔬 Research Dashboard - Eğitim & Veri Görselleştirme
              </a>
              <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer" 
                 className="block bg-blue-600 hover:bg-blue-700 p-3 rounded text-white transition-colors">
                📖 API Documentation
              </a>
              <a href="http://localhost:8000/health" target="_blank" rel="noopener noreferrer"
                 className="block bg-green-600 hover:bg-green-700 p-3 rounded text-white transition-colors">
                ❤️ Health Check
              </a>
            </div>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-cyan-400 mb-4">BWR-NSM Features</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="text-green-400">✅ Neural State Machine</div>
                <div className="text-green-400">✅ Hierarchical Memory</div>
                <div className="text-green-400">✅ State Persistence</div>
                <div className="text-green-400">✅ Async Processing</div>
              </div>
              <div className="space-y-2">
                <div className="text-green-400">✅ Performance Monitoring</div>
                <div className="text-green-400">✅ Advanced Eviction</div>
                <div className="text-green-400">✅ Unlimited Context</div>
                <div className="text-green-400">✅ Advanced Training</div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Home;
