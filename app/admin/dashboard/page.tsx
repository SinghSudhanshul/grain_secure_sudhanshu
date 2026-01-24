'use client';
export const dynamic = 'force-dynamic';

/**
 * GRAINSECURE ADMIN COMMAND CENTER
 * Premium Executive Dashboard with Real-time AI Insights
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Shield,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Users,
  Store,
  Activity,
  Brain,
  Network,
  Clock,
  Eye,
  ChevronRight,
  BarChart3,
  Zap,
  Target,
  RefreshCw,
  Bell,
  Settings,
  LogOut
} from 'lucide-react';
import { useSession, signOut } from 'next-auth/react';
import Link from 'next/link';
import { GlassPanel, LuxuryButton, MetricDisplay, StatusBadge, AnimatedCounter } from '@/components/ui/premium';

export default function AdminDashboard() {
  const { data: session } = useSession();
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  // Dashboard state
  const [metrics, setMetrics] = useState({
    totalBeneficiaries: 125847,
    totalShops: 2341,
    activeAlerts: 47,
    casesOpen: 12,
    leakageRate: 4.2,
    leakageTrend: -2.3,
    aiAccuracy: 97.3,
    transactionsToday: 8432
  });

  const [recentAlerts, setRecentAlerts] = useState([
    { id: 'ALT-001', title: 'Unusual velocity pattern detected', severity: 'critical', shop: 'FPS-234', time: '2 min ago' },
    { id: 'ALT-002', title: 'Stock discrepancy above threshold', severity: 'high', shop: 'FPS-891', time: '15 min ago' },
    { id: 'ALT-003', title: 'Geographic anomaly flagged', severity: 'medium', shop: 'FPS-456', time: '32 min ago' },
    { id: 'ALT-004', title: 'Multiple failed authentications', severity: 'high', shop: 'FPS-123', time: '45 min ago' },
    { id: 'ALT-005', title: 'Entitlement overclaim detected', severity: 'critical', shop: 'FPS-789', time: '1 hr ago' },
  ]);

  const [aiInsights, setAiInsights] = useState([
    { type: 'pattern', message: 'Increased fraud attempts in District 5 over the past 48 hours', confidence: 94 },
    { type: 'prediction', message: '3 shops flagged for potential inspection based on risk trajectory', confidence: 87 },
    { type: 'network', message: 'Possible collusion network detected involving 12 beneficiaries', confidence: 91 },
  ]);

  useEffect(() => {
    // Simulate loading
    setTimeout(() => setIsLoading(false), 1000);

    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      setLastUpdated(new Date());
      // Simulate metric updates
      setMetrics(prev => ({
        ...prev,
        transactionsToday: prev.transactionsToday + Math.floor(Math.random() * 10),
        activeAlerts: prev.activeAlerts + (Math.random() > 0.7 ? 1 : 0)
      }));
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#0A0A0F]">
      {/* ─────────────────────────────────────────────────────────────────────
          TOP NAVIGATION BAR
         ───────────────────────────────────────────────────────────────────── */}
      <nav className="sticky top-0 z-40 backdrop-blur-xl bg-black/80 border-b border-white/[0.05]">
        <div className="max-w-[1800px] mx-auto px-6 py-3">
          <div className="flex items-center justify-between">
            {/* Logo & Title */}
            <div className="flex items-center gap-6">
              <Link href="/" className="flex items-center gap-3">
                <Shield className="h-8 w-8 text-[#D4AF37]" />
                <span className="text-xl font-bold">
                  <span className="text-white">Grain</span>
                  <span className="text-[#D4AF37]">Secure</span>
                </span>
              </Link>
              <div className="h-6 w-px bg-white/10" />
              <h1 className="text-lg font-semibold text-white/80">Admin Command Center</h1>
            </div>

            {/* Right Side */}
            <div className="flex items-center gap-4">
              {/* Last Updated */}
              <div className="flex items-center gap-2 text-sm text-white/40">
                <Clock className="w-4 h-4" />
                <span>Updated {lastUpdated.toLocaleTimeString()}</span>
              </div>

              {/* Notifications */}
              <button className="relative p-2 rounded-lg hover:bg-white/5 transition-colors">
                <Bell className="w-5 h-5 text-white/60" />
                <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-red-500" />
              </button>

              {/* Settings */}
              <button className="p-2 rounded-lg hover:bg-white/5 transition-colors">
                <Settings className="w-5 h-5 text-white/60" />
              </button>

              {/* User Menu */}
              <div className="flex items-center gap-3 pl-4 border-l border-white/10">
                <div className="text-right">
                  <div className="text-sm font-medium text-white">{session?.user?.name || 'Admin'}</div>
                  <div className="text-xs text-white/40">{session?.user?.role || 'ADMIN'}</div>
                </div>
                <button
                  onClick={() => signOut({ callbackUrl: '/' })}
                  className="p-2 rounded-lg hover:bg-white/5 transition-colors text-white/60 hover:text-white"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* ─────────────────────────────────────────────────────────────────────
          MAIN CONTENT
         ───────────────────────────────────────────────────────────────────── */}
      <main className="max-w-[1800px] mx-auto px-6 py-8">
        {/* ─────────────────────────────────────────────────────────────────
            METRICS BAR
           ───────────────────────────────────────────────────────────────── */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4 mb-8">
          <MetricCard
            icon={<Users />}
            label="Beneficiaries"
            value={metrics.totalBeneficiaries}
            color="blue"
          />
          <MetricCard
            icon={<Store />}
            label="FPS Shops"
            value={metrics.totalShops}
            color="emerald"
          />
          <MetricCard
            icon={<AlertTriangle />}
            label="Active Alerts"
            value={metrics.activeAlerts}
            color="red"
            pulse
          />
          <MetricCard
            icon={<Target />}
            label="Open Cases"
            value={metrics.casesOpen}
            color="orange"
          />
          <MetricCard
            icon={<TrendingDown />}
            label="Leakage Rate"
            value={`${metrics.leakageRate}%`}
            trend={metrics.leakageTrend}
            color="purple"
          />
          <MetricCard
            icon={<Brain />}
            label="AI Accuracy"
            value={`${metrics.aiAccuracy}%`}
            color="gold"
          />
          <MetricCard
            icon={<Activity />}
            label="Today's Txns"
            value={metrics.transactionsToday}
            color="cyan"
          />
          <MetricCard
            icon={<Zap />}
            label="Response"
            value="50ms"
            color="green"
          />
        </div>

        {/* ─────────────────────────────────────────────────────────────────
            MAIN GRID
           ───────────────────────────────────────────────────────────────── */}
        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          {/* LEFT COLUMN - AI INSIGHTS */}
          <div className="lg:col-span-1 space-y-6">
            <GlassPanel variant="gold">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-[#D4AF37]/20 flex items-center justify-center">
                    <Brain className="w-5 h-5 text-[#D4AF37]" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">AI Insights</h2>
                    <p className="text-xs text-white/40">Real-time intelligence</p>
                  </div>
                </div>
                <StatusBadge status="success" label="Live" pulse />
              </div>

              <div className="space-y-4">
                {aiInsights.map((insight, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className="p-4 rounded-lg bg-white/[0.03] border border-white/[0.05] hover:border-[#D4AF37]/30 transition-colors"
                  >
                    <div className="flex items-start gap-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${insight.type === 'pattern' ? 'bg-blue-500/20 text-blue-400' :
                          insight.type === 'prediction' ? 'bg-purple-500/20 text-purple-400' :
                            'bg-orange-500/20 text-orange-400'
                        }`}>
                        {insight.type === 'pattern' ? <BarChart3 className="w-4 h-4" /> :
                          insight.type === 'prediction' ? <TrendingUp className="w-4 h-4" /> :
                            <Network className="w-4 h-4" />}
                      </div>
                      <div className="flex-1">
                        <p className="text-sm text-white/80">{insight.message}</p>
                        <div className="mt-2 flex items-center gap-2">
                          <div className="flex-1 h-1 rounded-full bg-white/10">
                            <div
                              className="h-full rounded-full bg-[#D4AF37]"
                              style={{ width: `${insight.confidence}%` }}
                            />
                          </div>
                          <span className="text-xs text-white/40">{insight.confidence}%</span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>

              <LuxuryButton variant="ghost" className="w-full mt-6">
                View All Insights
                <ChevronRight className="w-4 h-4" />
              </LuxuryButton>
            </GlassPanel>

            {/* Quick Actions */}
            <GlassPanel>
              <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
              <div className="grid grid-cols-2 gap-3">
                <ActionButton icon={<Eye />} label="View Alerts" href="/admin/alerts" />
                <ActionButton icon={<Network />} label="Network Map" href="/admin/network" />
                <ActionButton icon={<BarChart3 />} label="Analytics" href="/admin/analytics" />
                <ActionButton icon={<Settings />} label="Settings" href="/admin/settings" />
              </div>
            </GlassPanel>
          </div>

          {/* CENTER COLUMN - ALERTS */}
          <div className="lg:col-span-2">
            <GlassPanel className="h-full">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-red-500/20 flex items-center justify-center">
                    <AlertTriangle className="w-5 h-5 text-red-400" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-white">Active Alerts</h2>
                    <p className="text-xs text-white/40">{metrics.activeAlerts} requiring attention</p>
                  </div>
                </div>
                <LuxuryButton variant="ghost" size="sm">
                  <RefreshCw className="w-4 h-4" />
                  Refresh
                </LuxuryButton>
              </div>

              <div className="space-y-3">
                {recentAlerts.map((alert, idx) => (
                  <motion.div
                    key={alert.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className="flex items-center gap-4 p-4 rounded-xl bg-white/[0.02] border border-white/[0.05] hover:border-white/[0.1] hover:bg-white/[0.04] transition-all cursor-pointer group"
                  >
                    <StatusBadge
                      status={alert.severity as 'critical' | 'high' | 'medium' | 'low'}
                      pulse={alert.severity === 'critical'}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-white/40">{alert.id}</span>
                        <span className="text-xs text-[#D4AF37]">{alert.shop}</span>
                      </div>
                      <p className="text-sm text-white/80 truncate">{alert.title}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-white/40">{alert.time}</div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-white/20 group-hover:text-white/60 transition-colors" />
                  </motion.div>
                ))}
              </div>

              <div className="mt-6 flex items-center justify-between pt-6 border-t border-white/[0.05]">
                <span className="text-sm text-white/40">
                  Showing 5 of {metrics.activeAlerts} alerts
                </span>
                <LuxuryButton variant="outline" size="sm">
                  View All Alerts
                  <ChevronRight className="w-4 h-4" />
                </LuxuryButton>
              </div>
            </GlassPanel>
          </div>
        </div>

        {/* ─────────────────────────────────────────────────────────────────
            BOTTOM SECTION - STATS OVERVIEW
           ───────────────────────────────────────────────────────────────── */}
        <div className="grid md:grid-cols-3 gap-6">
          <StatCard
            title="Fraud Detection Rate"
            value={97.3}
            suffix="%"
            change={2.1}
            description="AI model performance this month"
            color="gold"
          />
          <StatCard
            title="Leakage Prevented"
            value={234.5}
            prefix="₹"
            suffix="Cr"
            change={15.4}
            description="Estimated savings this quarter"
            color="emerald"
          />
          <StatCard
            title="Investigation Time"
            value={2.3}
            suffix=" days"
            change={-18.2}
            description="Average case resolution time"
            color="blue"
          />
        </div>
      </main>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   HELPER COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */

function MetricCard({ icon, label, value, trend, color = 'white', pulse = false }: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  trend?: number;
  color?: string;
  pulse?: boolean;
}) {
  const colorClasses: Record<string, string> = {
    blue: 'from-blue-500/20 to-transparent border-blue-500/30 text-blue-400',
    emerald: 'from-emerald-500/20 to-transparent border-emerald-500/30 text-emerald-400',
    red: 'from-red-500/20 to-transparent border-red-500/30 text-red-400',
    orange: 'from-orange-500/20 to-transparent border-orange-500/30 text-orange-400',
    purple: 'from-purple-500/20 to-transparent border-purple-500/30 text-purple-400',
    gold: 'from-[#D4AF37]/20 to-transparent border-[#D4AF37]/30 text-[#D4AF37]',
    cyan: 'from-cyan-500/20 to-transparent border-cyan-500/30 text-cyan-400',
    green: 'from-green-500/20 to-transparent border-green-500/30 text-green-400',
    white: 'from-white/10 to-transparent border-white/20 text-white',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`relative p-4 rounded-xl bg-gradient-to-br border ${colorClasses[color]}`}
    >
      {pulse && (
        <span className="absolute top-2 right-2 flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
        </span>
      )}
      <div className="flex items-center gap-2 mb-2">
        <div className="w-6 h-6">{icon}</div>
      </div>
      <div className="text-2xl font-bold text-white">
        {typeof value === 'number' ? <AnimatedCounter value={value} /> : value}
      </div>
      <div className="text-xs text-white/50 mt-1">{label}</div>
      {trend !== undefined && (
        <div className={`text-xs mt-1 ${trend >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
          {trend >= 0 ? '↑' : '↓'} {Math.abs(trend)}%
        </div>
      )}
    </motion.div>
  );
}

function ActionButton({ icon, label, href }: { icon: React.ReactNode; label: string; href: string }) {
  return (
    <Link
      href={href}
      className="flex flex-col items-center gap-2 p-4 rounded-xl bg-white/[0.03] border border-white/[0.08] hover:bg-white/[0.06] hover:border-white/[0.15] transition-all group"
    >
      <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center text-white/60 group-hover:text-[#D4AF37] transition-colors">
        {icon}
      </div>
      <span className="text-xs text-white/60 group-hover:text-white transition-colors">{label}</span>
    </Link>
  );
}

function StatCard({ title, value, prefix = '', suffix = '', change, description, color }: {
  title: string;
  value: number;
  prefix?: string;
  suffix?: string;
  change: number;
  description: string;
  color: 'gold' | 'emerald' | 'blue';
}) {
  const colors = {
    gold: 'from-[#D4AF37]/10 to-transparent border-[#D4AF37]/20',
    emerald: 'from-emerald-500/10 to-transparent border-emerald-500/20',
    blue: 'from-blue-500/10 to-transparent border-blue-500/20',
  };

  return (
    <GlassPanel className={`bg-gradient-to-br ${colors[color]}`}>
      <h3 className="text-sm font-medium text-white/60 mb-4">{title}</h3>
      <div className="flex items-end gap-3 mb-2">
        <span className="text-4xl font-bold text-white">
          {prefix}<AnimatedCounter value={value} decimals={1} />{suffix}
        </span>
        <span className={`text-sm font-medium mb-1 ${change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
          {change >= 0 ? '↑' : '↓'} {Math.abs(change)}%
        </span>
      </div>
      <p className="text-sm text-white/40">{description}</p>
    </GlassPanel>
  );
}
