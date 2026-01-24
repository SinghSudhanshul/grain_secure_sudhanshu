'use client';

import { useState } from 'react';
import { signIn } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Shield, Lock, Mail, ArrowRight, Eye, EyeOff, Sparkles, AlertCircle } from 'lucide-react';
import { GlassPanel, LuxuryButton } from '@/components/ui/premium';

/* ═══════════════════════════════════════════════════════════════════════════
   PREMIUM LOGIN PAGE
   Harvey Specter Aesthetic | Split-screen Design
   ═══════════════════════════════════════════════════════════════════════════ */

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const result = await signIn('credentials', {
        redirect: false,
        email,
        password,
      });

      if (result?.error) {
        setError('Invalid credentials. Please try again.');
        setLoading(false);
        return;
      }

      // Fetch session to get role for routing
      const response = await fetch('/api/auth/session');
      const session = await response.json();

      if (session?.user?.role) {
        const roleRoutes: Record<string, string> = {
          ADMIN: '/admin/dashboard',
          INSPECTOR: '/inspector/dashboard',
          DEALER: '/dealer/dashboard',
          AUDITOR: '/auditor/dashboard',
          BENEFICIARY: '/beneficiary/portal',
        };

        router.push(roleRoutes[session.user.role] || '/');
      } else {
        router.push('/admin/dashboard');
      }
    } catch (err) {
      setError('An unexpected error occurred');
      setLoading(false);
    }
  };

  const quickLogin = (demoEmail: string, demoPassword: string) => {
    setEmail(demoEmail);
    setPassword(demoPassword);
  };

  const demoAccounts = [
    { role: 'Admin', email: 'admin@grainsecure.com', password: 'admin123', color: 'gold' },
    { role: 'Inspector', email: 'inspector@grainsecure.com', password: 'inspector123', color: 'blue' },
    { role: 'Dealer', email: 'dealer@grainsecure.com', password: 'dealer123', color: 'emerald' },
    { role: 'Auditor', email: 'auditor@grainsecure.com', password: 'auditor123', color: 'purple' },
    { role: 'Beneficiary', email: 'beneficiary@grainsecure.com', password: 'beneficiary123', color: 'cyan' },
  ];

  return (
    <div className="min-h-screen bg-[#0A0A0F] flex">
      {/* ─────────────────────────────────────────────────────────────────────
          LEFT SIDE - BRANDING & ANIMATION
         ───────────────────────────────────────────────────────────────────── */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0">
          {/* Gold gradient orb */}
          <motion.div
            className="absolute w-[600px] h-[600px] rounded-full opacity-20 blur-3xl"
            style={{
              background: 'radial-gradient(circle, #D4AF37 0%, transparent 70%)',
              left: '20%',
              top: '30%',
            }}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.15, 0.25, 0.15],
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />

          {/* Blue gradient orb */}
          <motion.div
            className="absolute w-[400px] h-[400px] rounded-full opacity-10 blur-3xl"
            style={{
              background: 'radial-gradient(circle, #3B82F6 0%, transparent 70%)',
              right: '10%',
              bottom: '20%',
            }}
            animate={{
              scale: [1, 1.3, 1],
              x: [0, 30, 0],
            }}
            transition={{
              duration: 10,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />

          {/* Grid overlay */}
          <div className="absolute inset-0 bg-grid opacity-20" />

          {/* Floating Particles */}
          {[...Array(15)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 rounded-full bg-white/30"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                y: [0, -50, 0],
                opacity: [0.2, 0.6, 0.2],
              }}
              transition={{
                duration: 4 + Math.random() * 3,
                repeat: Infinity,
                delay: Math.random() * 2,
              }}
            />
          ))}
        </div>

        {/* Content */}
        <div className="relative z-10 flex flex-col justify-center px-12 py-16">
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            className="flex items-center gap-3 mb-12"
          >
            <div className="relative">
              <Shield className="h-12 w-12 text-[#D4AF37]" />
              <div className="absolute inset-0 blur-lg bg-[#D4AF37]/40" />
            </div>
            <span className="text-3xl font-bold">
              <span className="text-white">Grain</span>
              <span className="text-[#D4AF37]">Secure</span>
            </span>
          </motion.div>

          {/* Tagline */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <h1 className="text-5xl font-bold leading-tight mb-6">
              <span className="text-gradient-white">Command</span>
              <br />
              <span className="text-gradient-gold">The Future of PDS</span>
            </h1>
            <p className="text-xl text-white/50 mb-12 max-w-md">
              Enterprise-grade AI monitoring protecting India's public distribution system.
            </p>
          </motion.div>

          {/* Features */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="space-y-4"
          >
            <FeatureItem icon={<Sparkles />} text="97.3% AI Detection Accuracy" />
            <FeatureItem icon={<Shield />} text="Real-time Fraud Prevention" />
            <FeatureItem icon={<Lock />} text="Tamper-proof Audit Trails" />
          </motion.div>
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────────────
          RIGHT SIDE - LOGIN FORM
         ───────────────────────────────────────────────────────────────────── */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="w-full max-w-md"
        >
          {/* Mobile Logo */}
          <div className="lg:hidden flex items-center gap-3 mb-8 justify-center">
            <Shield className="h-10 w-10 text-[#D4AF37]" />
            <span className="text-2xl font-bold">
              <span className="text-white">Grain</span>
              <span className="text-[#D4AF37]">Secure</span>
            </span>
          </div>

          <GlassPanel variant="elevated" className="p-8">
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-white mb-2">Welcome Back</h2>
              <p className="text-white/50">Enter your credentials to access the platform</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Email Input */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-white/70 flex items-center gap-2">
                  <Mail className="w-4 h-4" />
                  Email Address
                </label>
                <div className="relative">
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@grainsecure.com"
                    required
                    className="input-premium"
                  />
                </div>
              </div>

              {/* Password Input */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-white/70 flex items-center gap-2">
                  <Lock className="w-4 h-4" />
                  Password
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter your password"
                    required
                    className="input-premium pr-12"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-white/40 hover:text-white/70 transition-colors"
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm"
                >
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  {error}
                </motion.div>
              )}

              {/* Submit Button */}
              <LuxuryButton
                type="submit"
                variant="gold"
                size="lg"
                loading={loading}
                className="w-full"
              >
                {loading ? 'Signing In...' : 'Sign In'}
                <ArrowRight className="w-4 h-4" />
              </LuxuryButton>
            </form>

            {/* Divider */}
            <div className="flex items-center gap-4 my-8">
              <div className="flex-1 h-px bg-white/10" />
              <span className="text-xs text-white/30 uppercase tracking-wider">Demo Accounts</span>
              <div className="flex-1 h-px bg-white/10" />
            </div>

            {/* Quick Login Buttons */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {demoAccounts.map((account) => (
                <button
                  key={account.role}
                  type="button"
                  onClick={() => quickLogin(account.email, account.password)}
                  className={`px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200
                    bg-white/5 border border-white/10 text-white/70
                    hover:bg-white/10 hover:border-white/20 hover:text-white`}
                >
                  {account.role}
                </button>
              ))}
            </div>

            {/* Back Link */}
            <div className="mt-8 text-center">
              <Link
                href="/"
                className="text-sm text-white/40 hover:text-white/70 transition-colors inline-flex items-center gap-2"
              >
                ← Back to Home
              </Link>
            </div>
          </GlassPanel>

          {/* Footer */}
          <p className="mt-6 text-center text-xs text-white/30">
            Protected by enterprise-grade security • Smart India Hackathon 2026
          </p>
        </motion.div>
      </div>
    </div>
  );
}

function FeatureItem({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <div className="flex items-center gap-3 text-white/70">
      <div className="w-8 h-8 rounded-lg bg-[#D4AF37]/20 flex items-center justify-center text-[#D4AF37]">
        {icon}
      </div>
      <span>{text}</span>
    </div>
  );
}
