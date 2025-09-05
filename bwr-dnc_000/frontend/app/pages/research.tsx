import React, { useEffect, useState } from 'react';
import Head from 'next/head';
import dynamic from 'next/dynamic';

// Dynamic import to ensure client-only rendering (avoids SSR mismatch)
const AdvancedResearchDashboard = dynamic(() => import('../components/AdvancedResearchDashboard'), { ssr: false });

const ResearchPage: React.FC = () => {
  const [isClient, setIsClient] = useState(false);
  useEffect(() => { setIsClient(true); }, []);

  return (
    <div className="bg-gray-900 text-white min-h-screen">
      <Head>
        <title>BWR-DNC Advanced Research Dashboard</title>
        <meta name="description" content="Dynamic Neural Core vs Transformer Research Platform" />
      </Head>
      
      {/* Back to home link */}
      <div className="absolute top-4 left-4 z-50">
        <a 
          href="/" 
          className="text-cyan-400 hover:text-cyan-300 transition-colors text-sm flex items-center gap-2 bg-gray-800 px-3 py-2 rounded-lg border border-gray-700 hover:border-cyan-400"
        >
          <span>←</span> Back Home
        </a>
      </div>

      <div className="pt-16">
        {isClient ? (
          <AdvancedResearchDashboard />
        ) : (
          <div className="flex items-center justify-center min-h-screen">
            <div className="text-cyan-400 text-xl">Loading Advanced Research Dashboard…</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResearchPage;
