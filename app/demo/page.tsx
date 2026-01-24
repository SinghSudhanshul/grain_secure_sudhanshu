'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play,
  Cpu,
  Radio,
  Activity,
  ShieldCheck,
  AlertTriangle,
  Terminal,
  Database,
  CheckCircle2,
  RefreshCw
} from 'lucide-react';
import dynamic from 'next/dynamic';
import { LuxuryButton, GlassPanel, StatusBadge, AnimatedCounter } from '@/components/ui/premium';

const InteractiveBackground = dynamic(
  () => import('@/components/background/InteractiveBackground'),
  { ssr: false }
);

interface DemoResult {
  success: boolean;
  logs: string;
  isFallback?: boolean;
  metrics?: {
    accuracy: number;
    fraudDetected: number;
    valueSaved: string;
  };
}

export default function DemoPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [selectedModel, setSelectedModel] = useState('isolation_forest');
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<DemoResult | null>(null);

  const terminalRef = useRef<HTMLDivElement>(null);

  // Auto-scroll terminal
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  const runDemo = async () => {
    setIsRunning(true);
    setLogs(['> Initializing ML Environment...', '> Connecting to Secure Neural Engine...']);
    setProgress(10);
    setResult(null);

    try {
      // Simulate connection delay for realism
      await new Promise(r => setTimeout(r, 800));
      setLogs(prev => [...prev, `> Loading Model: ${selectedModel.toUpperCase()} v2.5`]);
      setProgress(30);

      const response = await fetch('/api/ml/run-demo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelName: selectedModel }),
      });

      const data = await response.json();

      // Stream simulated logs if using fallback or augment real logs
      if (data.logs) {
        const logLines = data.logs.split('\n');
        for (const line of logLines) {
          if (!line.trim()) continue;
          await new Promise(r => setTimeout(r, 50)); // Typing effect
          setLogs(prev => [...prev, line]);
          setProgress(prev => Math.min(prev + 5, 90));
        }
      }

      setResult(data);
      setProgress(100);
      setLogs(prev => [...prev, '> PROCESS COMPLETED SUCCESSFULLY', '> Generating Report...']);

    } catch (error) {
      setLogs(prev => [...prev, '> ERROR: Connection Interrupted', '> Retrying via redundant node...']);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0A0F] text-white overflow-x-hidden relative">
      <InteractiveBackground />

      {/* Overlay to darken background for better readability */}
      <div className="fixed inset-0 bg-[#0A0A0F]/80 z-[-5]" />

      <div className="max-w-7xl mx-auto px-6 py-12 relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-6 backdrop-blur-sm">
            <Cpu className="w-4 h-4 text-[#D4AF37]" />
            <span className="text-sm font-medium text-white/80">GrainSecure Neural Engine Direct Access</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-4 text-white">
            Live <span className="text-[#D4AF37]">AI Demonstration</span>
          </h1>
          <p className="text-xl text-white/50 max-w-2xl mx-auto">
            Experience our enterprise-grade fraud detection models running in real-time on synthetic PDS data.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

          {/* LEFT COLUMN: Controls & Config */}
          <div className="lg:col-span-4 space-y-6">
            <GlassPanel className="p-6">
              <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
                <Radio className="text-[#D4AF37]" /> Model Configuration
              </h2>

              <div className="space-y-4">
                <ModelOption
                  id="isolation_forest"
                  name="Isolation Forest"
                  desc="Unsupervised anomaly detection for novel fraud patterns"
                  active={selectedModel === 'isolation_forest'}
                  onClick={() => setSelectedModel('isolation_forest')}
                />
                <ModelOption
                  id="gradient_boosting"
                  name="Gradient Boosting"
                  desc="Supervised learning trained on confirmed fraud cases"
                  active={selectedModel === 'gradient_boosting'}
                  onClick={() => setSelectedModel('gradient_boosting')}
                />
                <ModelOption
                  id="autoencoder"
                  name="Deep Autoencoder"
                  desc="Neural network for complex non-linear pattern recognition"
                  active={selectedModel === 'autoencoder'}
                  onClick={() => setSelectedModel('autoencoder')}
                />
              </div>

              <div className="mt-8">
                <div className="flex justify-between text-sm mb-2 text-white/60">
                  <span>Data Source</span>
                  <span className="text-white">Synthetic PDS (100k)</span>
                </div>
                <div className="flex justify-between text-sm mb-6 text-white/60">
                  <span>Processing Node</span>
                  <span className="text-[#00FF94]">Active (v4.2)</span>
                </div>

                <LuxuryButton
                  onClick={runDemo}
                  disabled={isRunning}
                  variant="gold"
                  className="w-full justify-center text-lg py-6"
                >
                  {isRunning ? (
                    <span className="flex items-center gap-2">
                      <RefreshCw className="animate-spin w-5 h-5" /> Processing...
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      <Play className="fill-current w-5 h-5" /> Run Simulation
                    </span>
                  )}
                </LuxuryButton>
              </div>
            </GlassPanel>

            {/* System Status */}
            <GlassPanel className="p-6 bg-gradient-to-br from-[#0A0A0F] to-[#1a1a2e]">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-white/70 font-medium">System Load</h3>
                <Activity className="text-[#00FF94] w-5 h-5" />
              </div>
              <div className="h-32 flex items-end gap-1">
                {[...Array(20)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="flex-1 bg-[#D4AF37]/30 rounded-t-sm"
                    animate={{
                      height: isRunning ? `${Math.random() * 100}%` : '20%',
                      backgroundColor: isRunning ? '#D4AF37' : 'rgba(212, 175, 55, 0.2)'
                    }}
                    transition={{ duration: 0.5, repeat: Infinity, repeatType: "reverse", delay: i * 0.05 }}
                  />
                ))}
              </div>
            </GlassPanel>
          </div>

          {/* RIGHT COLUMN: Terminal & Results */}
          <div className="lg:col-span-8 space-y-6">

            {/* Terminal Window */}
            <GlassPanel className="p-0 overflow-hidden border-[#D4AF37]/20 flex flex-col h-[400px]">
              <div className="bg-[#111] px-4 py-2 border-b border-white/10 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-white/50" />
                  <span className="text-xs font-mono text-white/50">grainsecure-cli — v2.5.0</span>
                </div>
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-red-500/50" />
                  <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50" />
                  <div className="w-2.5 h-2.5 rounded-full bg-green-500/50" />
                </div>
              </div>

              <div
                ref={terminalRef}
                className="flex-1 p-6 font-mono text-sm overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10"
              >
                {!isRunning && logs.length === 0 && (
                  <div className="h-full flex flex-col items-center justify-center text-white/20">
                    <Database className="w-12 h-12 mb-4 opacity-50" />
                    <p>Ready to ingest data...</p>
                  </div>
                )}

                {logs.map((log, i) => (
                  <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    key={i}
                    className={`mb-2 ${log.includes('ERROR') ? 'text-red-400' : log.includes('SUCCESS') ? 'text-[#00FF94]' : log.includes('INFO') ? 'text-blue-400' : 'text-white/80'}`}
                  >
                    {log}
                  </motion.div>
                ))}

                {isRunning && (
                  <motion.div
                    animate={{ opacity: [0, 1, 0] }}
                    transition={{ duration: 0.8, repeat: Infinity }}
                    className="w-2 h-4 bg-[#D4AF37]"
                  />
                )}
              </div>

              {/* Progress Bar */}
              {isRunning && (
                <div className="h-1 bg-white/10 w-full overflow-hidden">
                  <motion.div
                    className="h-full bg-[#D4AF37]"
                    initial={{ width: "0%" }}
                    animate={{ width: `${progress}%` }}
                  />
                </div>
              )}
            </GlassPanel>

            {/* Results Section */}
            <AnimatePresence>
              {(result || isRunning) && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="grid grid-cols-1 md:grid-cols-3 gap-6"
                >
                  <MetricCard
                    label="Total Transactions"
                    value="100,000"
                    icon={<Database className="w-5 h-5" />}
                    delay={0}
                  />
                  <MetricCard
                    label="Anomalies Detected"
                    value={result?.metrics?.fraudDetected || "..."}
                    icon={<AlertTriangle className="text-amber-500 w-5 h-5" />}
                    delay={0.1}
                    highlight
                  />
                  <MetricCard
                    label="Value Protected"
                    value={result?.metrics?.valueSaved || "..."}
                    icon={<ShieldCheck className="text-[#00FF94] w-5 h-5" />}
                    delay={0.2}
                  />
                </motion.div>
              )}
            </AnimatePresence>

          </div>
        </div>
      </div>
    </div>
  );
}

function ModelOption({ id, name, desc, active, onClick }: { id: string, name: string, desc: string, active: boolean, onClick: () => void }) {
  return (
    <div
      onClick={onClick}
      className={`p-4 rounded-xl border transition-all cursor-pointer group ${active
          ? 'bg-[#D4AF37]/10 border-[#D4AF37] shadow-[0_0_15px_rgba(212,175,55,0.2)]'
          : 'bg-white/5 border-white/5 hover:bg-white/10 hover:border-white/20'
        }`}
    >
      <div className="flex items-center justify-between mb-1">
        <span className={`font-semibold ${active ? 'text-[#D4AF37]' : 'text-white'}`}>{name}</span>
        {active && <CheckCircle2 className="w-5 h-5 text-[#D4AF37]" />}
      </div>
      <p className="text-xs text-white/50 group-hover:text-white/70 transition-colors">{desc}</p>
    </div>
  );
}

function MetricCard({ label, value, icon, delay, highlight }: { label: string, value: string | number, icon: React.ReactNode, delay: number, highlight?: boolean }) {
  return (
    <GlassPanel className={`p-6 bg-[#0E0E12] ${highlight ? 'border-l-4 border-l-[#D4AF37]' : ''}`}>
      <div className="flex items-center gap-3 text-white/60 mb-2">
        {icon}
        <span className="text-sm font-medium">{label}</span>
      </div>
      <div className="text-3xl font-bold font-mono">
        {typeof value === 'number' ? (
          <AnimatedCounter value={value} />
        ) : (
          value
        )}
      </div>
    </GlassPanel>
  );
}
