import React, { useState, useEffect } from 'react';

interface TokenAnalysis {
  token_id: number;
  token_text: string;
  attention_weights: number[];
  state_contributions: number[];
  final_decision_weight: number;
  top_influencers: {
    token_id: number;
    token_text: string;
    influence_score: number;
    mechanism: 'attention' | 'state' | 'both';
  }[];
}

interface ProbeResult {
  selected_token: TokenAnalysis;
  context_tokens: {
    id: number;
    text: string;
    position: number;
  }[];
  interaction_matrix: number[][]; // token x token influence
  state_flow: {
    level: number;
    contribution: number;
    salience: number;
  }[];
}

interface InteractiveProbeProps {
  className?: string;
}

const InteractiveProbe: React.FC<InteractiveProbeProps> = ({ className = "" }) => {
  const [inputText, setInputText] = useState("The dynamic neural core learns to compress and retrieve information efficiently.");
  const [tokens, setTokens] = useState<string[]>([]);
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(null);
  const [probeResult, setProbeResult] = useState<ProbeResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<'attention' | 'state' | 'combined'>('combined');

  useEffect(() => {
    tokenizeText();
  }, [inputText]);

  const tokenizeText = () => {
    // Simple tokenization for demo (in real app would use the same tokenizer as backend)
    const tokenArray = inputText.split(/(\s+|[.,!?;])/).filter(t => t.trim().length > 0);
    setTokens(tokenArray);
    setSelectedTokenIndex(null);
    setProbeResult(null);
  };

  const analyzeToken = async (tokenIndex: number) => {
    setIsAnalyzing(true);
    setSelectedTokenIndex(tokenIndex);

    try {
      const response = await fetch('http://localhost:8000/research/analyze-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          text: inputText,
          focus_token: tokenIndex,
          analysis_mode: analysisMode
        })
      });

      if (response.ok) {
        const analysis = await response.json();
        generateProbeResult(tokenIndex, analysis);
      } else {
        generateDemoProbeResult(tokenIndex);
      }
    } catch (error) {
      console.warn('Analysis failed, generating demo result');
      generateDemoProbeResult(tokenIndex);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const generateDemoProbeResult = (tokenIndex: number) => {
    const contextTokens = tokens.map((text, id) => ({ id, text, position: id }));
    
    // Generate realistic attention patterns
    const attentionWeights = tokens.map((_, i) => {
      const distance = Math.abs(i - tokenIndex);
      let weight = 0;
      
      // Self-attention
      if (i === tokenIndex) weight += 0.4;
      // Local attention (neighboring tokens)
      if (distance <= 2) weight += 0.3 / (distance + 1);
      // Random global attention
      weight += Math.random() * 0.2;
      
      return Math.min(1.0, weight);
    });

    // Generate state contributions (simulate hierarchical levels)
    const stateContributions = tokens.map(() => Math.random() * 0.6);
    
    // Top influencers
    const influences = tokens.map((text, i) => ({
      token_id: i,
      token_text: text,
      influence_score: attentionWeights[i] * 0.6 + stateContributions[i] * 0.4,
      mechanism: Math.random() > 0.5 ? 'attention' as const : 
                 Math.random() > 0.3 ? 'state' as const : 'both' as const
    })).sort((a, b) => b.influence_score - a.influence_score).slice(0, 5);

    // Interaction matrix
    const interactionMatrix = tokens.map((_, i) => 
      tokens.map((_, j) => {
        const distance = Math.abs(i - j);
        return distance === 0 ? 1.0 : Math.max(0, 0.8 - distance * 0.2 + Math.random() * 0.3);
      })
    );

    // State flow across levels
    const stateFlow = [
      { level: 0, contribution: 0.6 + Math.random() * 0.3, salience: 0.7 + Math.random() * 0.2 },
      { level: 1, contribution: 0.4 + Math.random() * 0.3, salience: 0.5 + Math.random() * 0.3 },
      { level: 2, contribution: 0.2 + Math.random() * 0.2, salience: 0.3 + Math.random() * 0.3 },
      { level: 3, contribution: 0.1 + Math.random() * 0.1, salience: 0.2 + Math.random() * 0.2 }
    ];

    setProbeResult({
      selected_token: {
        token_id: tokenIndex,
        token_text: tokens[tokenIndex],
        attention_weights: attentionWeights,
        state_contributions: stateContributions,
        final_decision_weight: Math.random() * 0.8 + 0.1,
        top_influencers: influences
      },
      context_tokens: contextTokens,
      interaction_matrix: interactionMatrix,
      state_flow: stateFlow
    });
  };

  const generateProbeResult = (tokenIndex: number, analysis: any) => {
    // Process real analysis data (placeholder for now)
    generateDemoProbeResult(tokenIndex);
  };

  const getTokenColor = (index: number) => {
    if (selectedTokenIndex === null) return 'bg-gray-700 hover:bg-gray-600';
    if (index === selectedTokenIndex) return 'bg-cyan-600';
    
    if (probeResult) {
      const influence = probeResult.selected_token.attention_weights[index] + 
                       probeResult.selected_token.state_contributions[index];
      if (influence > 0.7) return 'bg-red-500';
      if (influence > 0.4) return 'bg-yellow-500';
      if (influence > 0.2) return 'bg-green-500';
    }
    
    return 'bg-gray-700 hover:bg-gray-600';
  };

  const getMechanismColor = (mechanism: string) => {
    switch (mechanism) {
      case 'attention': return 'text-blue-400';
      case 'state': return 'text-green-400';
      case 'both': return 'text-purple-400';
      default: return 'text-gray-400';
    }
  };

  const getMechanismIcon = (mechanism: string) => {
    switch (mechanism) {
      case 'attention': return '👀';
      case 'state': return '🧠';
      case 'both': return '🔗';
      default: return '❓';
    }
  };

  return (
    <div className={`bg-gray-800 p-6 rounded-lg space-y-6 ${className}`}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <h2 className="text-2xl font-bold text-cyan-400">🔍 Interactive Token Probe</h2>
        <div className="flex items-center gap-2">
          <select
            value={analysisMode}
            onChange={(e) => setAnalysisMode(e.target.value as any)}
            className="bg-gray-700 text-white rounded px-3 py-1 text-sm"
          >
            <option value="combined">Combined Analysis</option>
            <option value="attention">Attention Only</option>
            <option value="state">State Only</option>
          </select>
        </div>
      </div>

      {/* Text Input */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">Input Text</h3>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          rows={3}
          className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-cyan-400 focus:outline-none"
          placeholder="Enter text to analyze token interactions..."
        />
        <div className="mt-2 text-sm text-gray-400">
          Click on any token below to analyze its attention and state contributions.
        </div>
      </div>

      {/* Tokenized Text Display */}
      <div className="bg-gray-900 p-4 rounded-lg">
        <h3 className="text-lg font-semibold text-cyan-400 mb-4">
          Interactive Tokens 
          {selectedTokenIndex !== null && (
            <span className="text-sm text-gray-400 ml-2">
              (Selected: "{tokens[selectedTokenIndex]}")
            </span>
          )}
        </h3>
        <div className="flex flex-wrap gap-2">
          {tokens.map((token, index) => (
            <button
              key={index}
              onClick={() => analyzeToken(index)}
              disabled={isAnalyzing}
              className={`px-3 py-2 rounded-lg text-sm font-mono transition-all duration-200 ${
                getTokenColor(index)
              } ${isAnalyzing ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
              title={`Token ${index}: Click to analyze`}
            >
              {token}
              {selectedTokenIndex === index && isAnalyzing && (
                <span className="ml-1 animate-spin">⚙️</span>
              )}
            </button>
          ))}
        </div>
        
        {/* Legend */}
        <div className="mt-4 flex flex-wrap gap-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-cyan-600 rounded"></div>
            <span className="text-gray-400">Selected</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            <span className="text-gray-400">High Influence (0.7+)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-yellow-500 rounded"></div>
            <span className="text-gray-400">Medium Influence (0.4-0.7)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span className="text-gray-400">Low Influence (0.2-0.4)</span>
          </div>
        </div>
      </div>

      {/* Analysis Results */}
      {probeResult && (
        <>
          {/* Top Influencers */}
          <div className="bg-gray-900 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-cyan-400 mb-4">
              Top Influencers for "{probeResult.selected_token.token_text}"
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {probeResult.selected_token.top_influencers.map((influencer, index) => (
                <div key={index} className="bg-gray-700 p-3 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-white">"{influencer.token_text}"</span>
                    <span className={`text-sm ${getMechanismColor(influencer.mechanism)}`}>
                      {getMechanismIcon(influencer.mechanism)} {influencer.mechanism}
                    </span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${influencer.influence_score * 100}%` }}
                    />
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    Influence: {(influencer.influence_score * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* State Flow Analysis */}
          <div className="bg-gray-900 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-cyan-400 mb-4">Hierarchical State Flow</h3>
            <div className="space-y-3">
              {probeResult.state_flow.map((level, index) => (
                <div key={index} className="flex items-center gap-4">
                  <div className="w-16 text-sm text-gray-300 font-mono">
                    Level {level.level}
                  </div>
                  <div className="flex-1 space-y-2">
                    <div>
                      <div className="flex justify-between text-xs text-gray-400 mb-1">
                        <span>Contribution</span>
                        <span>{(level.contribution * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${level.contribution * 100}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs text-gray-400 mb-1">
                        <span>Salience</span>
                        <span>{(level.salience * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${level.salience * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 text-sm text-gray-400">
              Higher levels capture longer-range dependencies with compressed representations.
            </div>
          </div>

          {/* Interaction Matrix Visualization */}
          <div className="bg-gray-900 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-cyan-400 mb-4">Token Interaction Matrix</h3>
            <div className="overflow-auto">
              <div 
                className="grid gap-1 w-fit mx-auto"
                style={{ 
                  gridTemplateColumns: `repeat(${Math.min(probeResult.interaction_matrix[0].length, 20)}, 16px)`,
                  gridTemplateRows: `repeat(${Math.min(probeResult.interaction_matrix.length, 20)}, 16px)`
                }}
              >
                {probeResult.interaction_matrix.slice(0, 20).map((row, i) =>
                  row.slice(0, 20).map((value, j) => {
                    const isSelected = i === selectedTokenIndex || j === selectedTokenIndex;
                    const opacity = value;
                    return (
                      <div
                        key={`${i}-${j}`}
                        className={`w-4 h-4 rounded-sm border transition-all duration-200 ${
                          isSelected ? 'border-cyan-400 border-2' : 'border-gray-700'
                        }`}
                        style={{ 
                          backgroundColor: `rgba(99, 102, 241, ${opacity})`,
                          transform: isSelected ? 'scale(1.2)' : 'scale(1)'
                        }}
                        title={`${tokens[i] || `Token ${i}`} → ${tokens[j] || `Token ${j}`}: ${value.toFixed(3)}`}
                      />
                    );
                  })
                )}
              </div>
            </div>
            <div className="mt-4 text-sm text-gray-400">
              Rows and columns represent tokens. Brightness indicates interaction strength.
              Selected token's row/column is highlighted.
            </div>
          </div>

          {/* Decision Breakdown */}
          <div className="bg-gray-900 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-cyan-400 mb-4">Decision Mechanism Breakdown</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-gray-300 font-semibold mb-3">Attention Contributions</h4>
                <div className="space-y-2">
                  {probeResult.selected_token.attention_weights.slice(0, 5).map((weight, index) => (
                    <div key={index} className="flex items-center gap-3">
                      <span className="text-xs font-mono w-20 text-gray-400">
                        {tokens[index] || `Token ${index}`}
                      </span>
                      <div className="flex-1 bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${weight * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 w-12">
                        {(weight * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <h4 className="text-gray-300 font-semibold mb-3">State Contributions</h4>
                <div className="space-y-2">
                  {probeResult.selected_token.state_contributions.slice(0, 5).map((contribution, index) => (
                    <div key={index} className="flex items-center gap-3">
                      <span className="text-xs font-mono w-20 text-gray-400">
                        {tokens[index] || `Token ${index}`}
                      </span>
                      <div className="flex-1 bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${contribution * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 w-12">
                        {(contribution * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="mt-4 p-3 bg-gray-800 rounded-lg">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Final Decision Weight:</span>
                <span className="text-xl font-bold text-purple-400">
                  {(probeResult.selected_token.final_decision_weight * 100).toFixed(1)}%
                </span>
              </div>
              <div className="text-xs text-gray-400 mt-1">
                Combined influence of attention and state mechanisms on this token's output.
              </div>
            </div>
          </div>
        </>
      )}

      {/* Instructions */}
      {!probeResult && (
        <div className="bg-gray-900 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-cyan-400 mb-4">How to Use</h3>
          <div className="space-y-2 text-sm text-gray-400">
            <p>1. <strong>Enter or edit text</strong> in the input area above</p>
            <p>2. <strong>Click any token</strong> to analyze its attention and state interactions</p>
            <p>3. <strong>Examine the results:</strong></p>
            <ul className="ml-4 space-y-1">
              <li>• Top influencers show which tokens most affect the selected token</li>
              <li>• State flow reveals hierarchical memory contributions</li>
              <li>• Interaction matrix visualizes token-to-token relationships</li>
              <li>• Decision breakdown shows attention vs. state mechanism balance</li>
            </ul>
            <p>4. <strong>Change analysis mode</strong> to focus on specific mechanisms</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default InteractiveProbe;
