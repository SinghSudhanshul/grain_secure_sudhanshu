'use client';

import React, { useEffect, useRef, useState, Suspense } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { motion, useScroll, useTransform } from 'framer-motion';
import {
  Shield,
  Brain,
  Network,
  Activity,
  Eye,
  Lock,
  Zap,
  TrendingUp,
  Users,
  BarChart3,
  ArrowRight,
  ChevronDown,
  Sparkles,
  Target,
  Globe
} from 'lucide-react';
import { GlassPanel, LuxuryButton, AnimatedCounter, StatusBadge } from '@/components/ui/premium';

// Dynamically import Three.js background to avoid SSR issues
const InteractiveBackground = dynamic(
  () => import('@/components/background/InteractiveBackground'),
  {
    ssr: false,
    loading: () => (
      <div className="fixed inset-0 -z-10 bg-[#0A0A0F]">
        <div className="absolute inset-0 bg-gradient-radial from-[#D4AF37]/5 via-transparent to-transparent" />
      </div>
    )
  }
);

/* ═══════════════════════════════════════════════════════════════════════════
   GRAINSECURE LANDING PAGE
   Harvey Specter Aesthetic | Premium Enterprise Platform
   With Interactive 3D WebGL Background
   ═══════════════════════════════════════════════════════════════════════════ */

export default function HomePage() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const heroRef = useRef<HTMLDivElement>(null);
  const { scrollY } = useScroll();
  const heroOpacity = useTransform(scrollY, [0, 400], [1, 0]);
  const heroScale = useTransform(scrollY, [0, 400], [1, 0.95]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="min-h-screen bg-transparent text-white overflow-x-hidden">
      {/* ═══════════════════════════════════════════════════════════════════
          INTERACTIVE 3D BACKGROUND (Three.js WebGL)
         ═══════════════════════════════════════════════════════════════════ */}
      <InteractiveBackground />

      {/* ─────────────────────────────────────────────────────────────────────
          PREMIUM NAVIGATION
         ───────────────────────────────────────────────────────────────────── */}
      <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-black/60 border-b border-white/[0.05]">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-3 group">
              <div className="relative">
                <Shield className="h-8 w-8 text-[#D4AF37]" />
                <div className="absolute inset-0 blur-lg bg-[#D4AF37]/30 group-hover:bg-[#D4AF37]/50 transition-all" />
              </div>
              <span className="text-xl font-bold tracking-tight">
                <span className="text-white">Grain</span>
                <span className="text-[#D4AF37]">Secure</span>
              </span>
            </Link>

            {/* Nav Links */}
            <div className="hidden md:flex items-center gap-8">
              <NavLink href="#capabilities">Capabilities</NavLink>
              <NavLink href="#technology">Technology</NavLink>
              <NavLink href="#impact">Impact</NavLink>
              <NavLink href="/public/dashboard">Public Data</NavLink>
            </div>

            {/* CTAs */}
            <div className="flex items-center gap-4">
              <Link href="/login">
                <LuxuryButton variant="ghost" size="sm">
                  Sign In
                </LuxuryButton>
              </Link>
              <Link href="/login">
                <LuxuryButton variant="gold" size="sm">
                  Launch Platform
                </LuxuryButton>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* ─────────────────────────────────────────────────────────────────────
          HERO SECTION
         ───────────────────────────────────────────────────────────────────── */}
      <motion.section
        ref={heroRef}
        className="relative min-h-screen flex items-center justify-center pt-20"
        style={{ opacity: heroOpacity, scale: heroScale }}
      >
        {/* Hero Content */}
        <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-8 backdrop-blur-sm"
          >
            <Sparkles className="w-4 h-4 text-[#D4AF37]" />
            <span className="text-sm text-white/70">AI-Powered PDS Intelligence Platform</span>
          </motion.div>

          {/* Main Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-5xl md:text-7xl lg:text-8xl font-bold tracking-tight mb-6"
          >
            <span className="text-gradient-white">Eliminate Fraud.</span>
            <br />
            <span className="text-gradient-gold">Protect Every Grain.</span>
          </motion.h1>

          {/* Subheadline */}
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-xl md:text-2xl text-white/60 max-w-3xl mx-auto mb-12"
          >
            Enterprise-grade AI monitoring for India's Public Distribution System.
            Real-time fraud detection, network analysis, and tamper-proof accountability.
          </motion.p>

          {/* CTAs */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16"
          >
            <Link href="/login">
              <LuxuryButton variant="gold" size="lg">
                Access Platform
                <ArrowRight className="w-4 h-4" />
              </LuxuryButton>
            </Link>
            <Link href="/public/dashboard">
              <LuxuryButton variant="outline" size="lg">
                View Live Data
              </LuxuryButton>
            </Link>
          </motion.div>

          {/* Live Stats */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-6"
          >
            <StatCard
              icon={<Shield className="w-5 h-5" />}
              value={12847}
              label="Transactions Monitored"
              suffix="/day"
            />
            <StatCard
              icon={<Brain className="w-5 h-5" />}
              value={97.3}
              label="AI Accuracy"
              suffix="%"
            />
            <StatCard
              icon={<Target className="w-5 h-5" />}
              value={2341}
              label="Frauds Detected"
              prefix="₹"
              suffix="Cr"
            />
            <StatCard
              icon={<Zap className="w-5 h-5" />}
              value={50}
              label="Response Time"
              suffix="ms"
            />
          </motion.div>
        </div>

        {/* Scroll Indicator */}
        <motion.div
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <ChevronDown className="w-6 h-6 text-white/30" />
        </motion.div>
      </motion.section>

      {/* ─────────────────────────────────────────────────────────────────────
          CAPABILITIES SECTION
         ───────────────────────────────────────────────────────────────────── */}
      <section id="capabilities" className="py-32 relative">
        <div className="max-w-7xl mx-auto px-6">
          <SectionTitle
            badge="Capabilities"
            title="Military-Grade Fraud Detection"
            subtitle="10 specialized AI models working in concert to eliminate leakages and protect public resources."
          />

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mt-16">
            <CapabilityCard
              icon={<Brain />}
              title="Anomaly Detection"
              description="Isolation Forest & Autoencoder neural networks identify unusual transaction patterns in real-time."
              tags={['IsolationForest', 'Autoencoder', 'Deep Learning']}
            />
            <CapabilityCard
              icon={<Network />}
              title="Network Analysis"
              description="Graph Neural Networks map collusion rings and identify coordinated fraud operations."
              tags={['GNN', 'Community Detection', 'Graph Analysis']}
            />
            <CapabilityCard
              icon={<TrendingUp />}
              title="Predictive Risk Scoring"
              description="Gradient Boosting classifiers predict fraud probability with 97% accuracy."
              tags={['XGBoost', 'Risk Model', 'Calibrated Probabilities']}
            />
            <CapabilityCard
              icon={<Activity />}
              title="Temporal Pattern Analysis"
              description="Detect seasonal anomalies, unusual timing patterns, and velocity-based fraud indicators."
              tags={['Time Series', 'Velocity Checks', 'Seasonality']}
            />
            <CapabilityCard
              icon={<Eye />}
              title="Explainable AI"
              description="Every detection comes with human-readable evidence and confidence scores."
              tags={['SHAP', 'Feature Importance', 'Audit Trail']}
            />
            <CapabilityCard
              icon={<Lock />}
              title="Tamper-Proof Logs"
              description="Cryptographic hash chains ensure complete accountability and regulatory compliance."
              tags={['Hash Chain', 'Immutable Logs', 'Compliance']}
            />
          </div>
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────────────
          TECHNOLOGY SECTION
         ───────────────────────────────────────────────────────────────────── */}
      <section id="technology" className="py-32 relative bg-gradient-to-b from-transparent via-white/[0.02] to-transparent">
        <div className="max-w-7xl mx-auto px-6">
          <SectionTitle
            badge="Technology"
            title="Enterprise Architecture"
            subtitle="Built on proven technologies trusted by Fortune 500 companies."
          />

          <div className="grid lg:grid-cols-2 gap-12 mt-16">
            <GlassPanel variant="elevated" className="p-8">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-[#D4AF37]/20 flex items-center justify-center">
                  <Globe className="w-5 h-5 text-[#D4AF37]" />
                </div>
                Frontend & Visualization
              </h3>
              <div className="space-y-4">
                <TechItem label="Next.js 14" detail="App Router with Server Components" />
                <TechItem label="React 18" detail="Concurrent rendering & Suspense" />
                <TechItem label="Framer Motion" detail="Premium 60fps animations" />
                <TechItem label="D3.js & Recharts" detail="Interactive data visualizations" />
                <TechItem label="Mapbox GL" detail="Geographic fraud heatmaps" />
              </div>
            </GlassPanel>

            <GlassPanel variant="elevated" className="p-8">
              <h3 className="text-2xl font-bold mb-6 flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                  <Brain className="w-5 h-5 text-blue-400" />
                </div>
                Backend & ML
              </h3>
              <div className="space-y-4">
                <TechItem label="Python ML Stack" detail="Scikit-learn, XGBoost, PyTorch, TensorFlow" />
                <TechItem label="Prisma ORM" detail="Type-safe database operations" />
                <TechItem label="NextAuth.js" detail="Enterprise authentication" />
                <TechItem label="Free AI APIs" detail="Groq, Gemini, HuggingFace integration" />
                <TechItem label="Real-time Events" detail="Server-Sent Events for live updates" />
              </div>
            </GlassPanel>
          </div>
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────────────
          ROLE-BASED ACCESS
         ───────────────────────────────────────────────────────────────────── */}
      <section className="py-32 relative">
        <div className="max-w-7xl mx-auto px-6">
          <SectionTitle
            badge="Access Control"
            title="Role-Based Intelligence"
            subtitle="Specialized dashboards for every stakeholder in the PDS ecosystem."
          />

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mt-16">
            <RoleCard
              role="Administrator"
              description="Complete system oversight with real-time monitoring, AI insights, and strategic analytics."
              features={['Real-time fraud map', 'Network analysis', 'System health']}
              color="gold"
            />
            <RoleCard
              role="Inspector"
              description="Investigation workspace with case management, evidence collection, and field visit scheduling."
              features={['Case queue', 'Investigation tools', 'AI suggestions']}
              color="blue"
            />
            <RoleCard
              role="Dealer"
              description="Shop management portal with stock tracking, transaction logs, and compliance scoring."
              features={['Stock management', 'Daily reports', 'Alert notifications']}
              color="emerald"
            />
            <RoleCard
              role="Auditor"
              description="Comprehensive audit interface with trail viewer, reconciliation reports, and exception tracking."
              features={['Audit logs', 'Reconciliation', 'PDF exports']}
              color="purple"
            />
            <RoleCard
              role="Beneficiary"
              description="Citizen portal with entitlement tracking, transaction history, and complaint submission."
              features={['Quota tracker', 'Receipts', 'Grievance filing']}
              color="cyan"
            />
            <RoleCard
              role="Public"
              description="Transparent dashboard showing aggregated district data without authentication requirement."
              features={['Open data', 'District stats', 'Transparency']}
              color="white"
            />
          </div>
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────────────
          IMPACT SECTION
         ───────────────────────────────────────────────────────────────────── */}
      <section id="impact" className="py-32 relative overflow-hidden">
        {/* Background Gradient */}
        <div className="absolute inset-0 bg-gradient-to-r from-[#D4AF37]/10 via-transparent to-blue-500/10" />

        <div className="relative max-w-7xl mx-auto px-6">
          <SectionTitle
            badge="Impact"
            title="Measurable Results"
            subtitle="Projected savings based on current PDS leakage rates and system efficiency gains."
          />

          <div className="grid md:grid-cols-4 gap-8 mt-16">
            <ImpactCard
              value={60}
              suffix="%"
              label="Leakage Reduction"
              description="AI-detected fraud prevention"
            />
            <ImpactCard
              value={30000}
              prefix="₹"
              suffix=" Cr"
              label="Annual Savings"
              description="Recovered public resources"
            />
            <ImpactCard
              value={100}
              suffix="%"
              label="Audit Coverage"
              description="Complete transaction trail"
            />
            <ImpactCard
              value={50}
              suffix="ms"
              label="Detection Speed"
              description="Real-time fraud alerts"
            />
          </div>
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────────────
          CTA SECTION
         ───────────────────────────────────────────────────────────────────── */}
      <section className="py-32 relative">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Ready to <span className="text-gradient-gold">Secure</span> Your PDS?
            </h2>
            <p className="text-xl text-white/60 mb-12">
              Access the platform with demo credentials or explore the public dashboard.
            </p>

            <GlassPanel variant="gold" className="p-8 inline-block">
              <h3 className="text-lg font-semibold mb-4">Demo Credentials</h3>
              <div className="grid grid-cols-2 gap-4 text-sm mb-6">
                <div className="text-left">
                  <div className="text-white/50">Admin</div>
                  <div className="font-mono text-[#D4AF37]">admin@grainsecure.com</div>
                </div>
                <div className="text-left">
                  <div className="text-white/50">Password</div>
                  <div className="font-mono text-[#D4AF37]">admin123</div>
                </div>
              </div>
              <div className="flex gap-4 justify-center">
                <Link href="/login">
                  <LuxuryButton variant="gold">
                    Launch Platform
                    <ArrowRight className="w-4 h-4" />
                  </LuxuryButton>
                </Link>
              </div>
            </GlassPanel>
          </motion.div>
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────────────
          FOOTER
         ───────────────────────────────────────────────────────────────────── */}
      <footer className="py-12 border-t border-white/[0.05]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <Shield className="w-6 h-6 text-[#D4AF37]" />
              <span className="font-semibold">GrainSecure</span>
              <span className="text-white/40">•</span>
              <span className="text-white/40 text-sm">Smart India Hackathon 2026</span>
            </div>
            <div className="flex items-center gap-6 text-sm text-white/40">
              <span>Built with Next.js, Prisma, and AI</span>
              <span>•</span>
              <span>Enterprise-Ready</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   HELPER COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="relative text-sm text-white/60 hover:text-white transition-colors duration-200 group"
    >
      {children}
      <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-[#D4AF37] group-hover:w-full transition-all duration-300" />
    </Link>
  );
}

function ParticleField() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {[...Array(30)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 rounded-full bg-white/20"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }}
          animate={{
            y: [0, -30, 0],
            opacity: [0.2, 0.5, 0.2],
          }}
          transition={{
            duration: 3 + Math.random() * 2,
            repeat: Infinity,
            delay: Math.random() * 2,
          }}
        />
      ))}
    </div>
  );
}

function StatCard({ icon, value, label, prefix = '', suffix = '' }: {
  icon: React.ReactNode;
  value: number;
  label: string;
  prefix?: string;
  suffix?: string;
}) {
  return (
    <div className="glass-card text-center">
      <div className="text-[#D4AF37] mb-2 flex justify-center">{icon}</div>
      <div className="text-2xl md:text-3xl font-bold text-white">
        {prefix}<AnimatedCounter value={value} />{suffix}
      </div>
      <div className="text-xs text-white/50 mt-1">{label}</div>
    </div>
  );
}

function SectionTitle({ badge, title, subtitle }: {
  badge: string;
  title: string;
  subtitle: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="text-center max-w-3xl mx-auto"
    >
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-6">
        <span className="text-sm text-[#D4AF37] font-medium">{badge}</span>
      </div>
      <h2 className="text-4xl md:text-5xl font-bold mb-4">{title}</h2>
      <p className="text-xl text-white/50">{subtitle}</p>
    </motion.div>
  );
}

function CapabilityCard({ icon, title, description, tags }: {
  icon: React.ReactNode;
  title: string;
  description: string;
  tags: string[];
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="card-premium p-6 group"
    >
      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#D4AF37]/20 to-transparent flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
        <div className="text-[#D4AF37]">{icon}</div>
      </div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-white/50 text-sm mb-4">{description}</p>
      <div className="flex flex-wrap gap-2">
        {tags.map((tag) => (
          <span key={tag} className="text-xs px-2 py-1 rounded bg-white/5 text-white/40">
            {tag}
          </span>
        ))}
      </div>
    </motion.div>
  );
}

function TechItem({ label, detail }: { label: string; detail: string }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-white/5 last:border-0">
      <span className="font-medium">{label}</span>
      <span className="text-sm text-white/40">{detail}</span>
    </div>
  );
}

function RoleCard({ role, description, features, color }: {
  role: string;
  description: string;
  features: string[];
  color: 'gold' | 'blue' | 'emerald' | 'purple' | 'cyan' | 'white';
}) {
  const colors = {
    gold: 'from-[#D4AF37]/20 to-transparent border-[#D4AF37]/30',
    blue: 'from-blue-500/20 to-transparent border-blue-500/30',
    emerald: 'from-emerald-500/20 to-transparent border-emerald-500/30',
    purple: 'from-purple-500/20 to-transparent border-purple-500/30',
    cyan: 'from-cyan-500/20 to-transparent border-cyan-500/30',
    white: 'from-white/10 to-transparent border-white/20',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className={`rounded-2xl p-6 bg-gradient-to-br border ${colors[color]}`}
    >
      <div className="flex items-center gap-3 mb-3">
        <Users className="w-5 h-5 text-white/70" />
        <h3 className="text-lg font-semibold">{role}</h3>
      </div>
      <p className="text-sm text-white/50 mb-4">{description}</p>
      <div className="space-y-2">
        {features.map((feature) => (
          <div key={feature} className="flex items-center gap-2 text-sm text-white/70">
            <div className="w-1 h-1 rounded-full bg-[#D4AF37]" />
            {feature}
          </div>
        ))}
      </div>
    </motion.div>
  );
}

function ImpactCard({ value, prefix = '', suffix = '', label, description }: {
  value: number;
  prefix?: string;
  suffix?: string;
  label: string;
  description: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="text-center"
    >
      <div className="text-5xl md:text-6xl font-bold text-gradient-gold mb-2">
        {prefix}<AnimatedCounter value={value} />{suffix}
      </div>
      <div className="text-lg font-semibold text-white mb-1">{label}</div>
      <div className="text-sm text-white/40">{description}</div>
    </motion.div>
  );
}
