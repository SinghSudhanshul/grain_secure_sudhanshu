'use client';

import React, { forwardRef } from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { cn } from '@/lib/utils';

/* ═══════════════════════════════════════════════════════════════════════════
   PREMIUM GLASS PANEL
   Elevated glassmorphism container with optional glow effects
   ═══════════════════════════════════════════════════════════════════════════ */

interface GlassPanelProps extends HTMLMotionProps<'div'> {
  variant?: 'default' | 'subtle' | 'elevated' | 'gold';
  glow?: boolean;
  hover?: boolean;
}

export const GlassPanel = forwardRef<HTMLDivElement, GlassPanelProps>(
  ({ className, variant = 'default', glow = false, hover = true, children, ...props }, ref) => {
    const variants = {
      default: 'bg-white/[0.03] border-white/[0.08] backdrop-blur-xl',
      subtle: 'bg-white/[0.02] border-white/[0.05] backdrop-blur-md',
      elevated: 'bg-gradient-to-br from-white/[0.08] to-white/[0.02] border-white/[0.12] backdrop-blur-2xl',
      gold: 'bg-gradient-to-br from-[#D4AF37]/10 to-transparent border-[#D4AF37]/20 backdrop-blur-xl',
    };

    return (
      <motion.div
        ref={ref}
        className={cn(
          'relative rounded-2xl border p-6',
          'shadow-xl shadow-black/20',
          variants[variant],
          hover && 'transition-all duration-300 hover:border-white/20 hover:bg-white/[0.05]',
          glow && 'shadow-[0_0_40px_-10px_rgba(212,175,55,0.2)]',
          className
        )}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
        {...props}
      >
        {/* Subtle inner glow */}
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-b from-white/[0.02] to-transparent pointer-events-none" />
        <div className="relative z-10">{children}</div>
      </motion.div>
    );
  }
);
GlassPanel.displayName = 'GlassPanel';

/* ═══════════════════════════════════════════════════════════════════════════
   LUXURY BUTTON
   Premium gold-accented CTAs with hover effects
   ═══════════════════════════════════════════════════════════════════════════ */

interface LuxuryButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'gold' | 'ghost' | 'outline' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
}

export const LuxuryButton = forwardRef<HTMLButtonElement, LuxuryButtonProps>(
  ({ className, variant = 'gold', size = 'md', loading, icon, children, disabled, ...props }, ref) => {
    const variants = {
      gold: 'bg-gradient-to-r from-[#D4AF37] via-[#F4D03F] to-[#D4AF37] text-black font-semibold shadow-lg shadow-[#D4AF37]/25 hover:shadow-xl hover:shadow-[#D4AF37]/30',
      ghost: 'bg-white/5 border border-white/10 text-white/80 hover:bg-white/10 hover:text-white hover:border-white/20',
      outline: 'bg-transparent border-2 border-[#D4AF37] text-[#D4AF37] hover:bg-[#D4AF37]/10',
      danger: 'bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20',
    };

    const sizes = {
      sm: 'px-4 py-2 text-xs',
      md: 'px-6 py-3 text-sm',
      lg: 'px-8 py-4 text-base',
    };

    return (
      <motion.button
        ref={ref}
        className={cn(
          'relative inline-flex items-center justify-center gap-2',
          'rounded-lg font-medium tracking-wide uppercase',
          'transition-all duration-300',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          variants[variant],
          sizes[size],
          className
        )}
        whileHover={{ scale: disabled ? 1 : 1.02 }}
        whileTap={{ scale: disabled ? 1 : 0.98 }}
        disabled={disabled || loading}
        {...props}
      >
        {loading ? (
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        ) : icon ? (
          <span className="w-4 h-4">{icon}</span>
        ) : null}
        {children}
      </motion.button>
    );
  }
);
LuxuryButton.displayName = 'LuxuryButton';

/* ═══════════════════════════════════════════════════════════════════════════
   METRIC DISPLAY
   Animated counter with trend indicator
   ═══════════════════════════════════════════════════════════════════════════ */

interface MetricDisplayProps {
  label: string;
  value: number | string;
  prefix?: string;
  suffix?: string;
  trend?: number;
  trendLabel?: string;
  icon?: React.ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'danger';
  size?: 'sm' | 'md' | 'lg';
}

export const MetricDisplay: React.FC<MetricDisplayProps> = ({
  label,
  value,
  prefix,
  suffix,
  trend,
  trendLabel,
  icon,
  variant = 'default',
  size = 'md',
}) => {
  const variantColors = {
    default: 'text-white',
    success: 'text-emerald-400',
    warning: 'text-amber-400',
    danger: 'text-red-400',
  };

  const sizes = {
    sm: { value: 'text-2xl', label: 'text-xs' },
    md: { value: 'text-4xl', label: 'text-sm' },
    lg: { value: 'text-5xl', label: 'text-base' },
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-white/60">
        {icon && <span className="w-4 h-4">{icon}</span>}
        <span className={cn('font-medium uppercase tracking-wider', sizes[size].label)}>
          {label}
        </span>
      </div>
      <div className="flex items-baseline gap-2">
        <motion.span
          className={cn('font-bold tracking-tight', sizes[size].value, variantColors[variant])}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          {prefix}
          {typeof value === 'number' ? value.toLocaleString() : value}
          {suffix}
        </motion.span>
        {trend !== undefined && (
          <span
            className={cn(
              'text-sm font-medium',
              trend >= 0 ? 'text-emerald-400' : 'text-red-400'
            )}
          >
            {trend >= 0 ? '↑' : '↓'} {Math.abs(trend)}%
            {trendLabel && <span className="text-white/40 ml-1">{trendLabel}</span>}
          </span>
        )}
      </div>
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════════════
   STATUS BADGE
   Risk level indicators with pulse animation
   ═══════════════════════════════════════════════════════════════════════════ */

interface StatusBadgeProps {
  status: 'critical' | 'high' | 'medium' | 'low' | 'info' | 'success';
  label?: string;
  pulse?: boolean;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  label,
  pulse = false,
  size = 'md',
}) => {
  const statusConfig = {
    critical: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30', dot: 'bg-red-500' },
    high: { bg: 'bg-orange-500/20', text: 'text-orange-400', border: 'border-orange-500/30', dot: 'bg-orange-500' },
    medium: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/30', dot: 'bg-yellow-500' },
    low: { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/30', dot: 'bg-green-500' },
    info: { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30', dot: 'bg-blue-500' },
    success: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30', dot: 'bg-emerald-500' },
  };

  const config = statusConfig[status];
  const displayLabel = label || status.charAt(0).toUpperCase() + status.slice(1);

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border',
        config.bg,
        config.text,
        config.border,
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-xs font-medium'
      )}
    >
      <span className="relative flex h-2 w-2">
        {pulse && (
          <span
            className={cn(
              'absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping',
              config.dot
            )}
          />
        )}
        <span className={cn('relative inline-flex rounded-full h-2 w-2', config.dot)} />
      </span>
      {displayLabel}
    </span>
  );
};

/* ═══════════════════════════════════════════════════════════════════════════
   ANIMATED COUNTER
   Smooth number counting animation
   ═══════════════════════════════════════════════════════════════════════════ */

interface AnimatedCounterProps {
  value: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
  decimals?: number;
}

export const AnimatedCounter: React.FC<AnimatedCounterProps> = ({
  value,
  duration = 2,
  prefix = '',
  suffix = '',
  className,
  decimals = 0,
}) => {
  const [displayValue, setDisplayValue] = React.useState(0);

  React.useEffect(() => {
    let startTime: number;
    let animationFrame: number;

    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / (duration * 1000), 1);

      // Easing function for smooth animation
      const easeOutExpo = 1 - Math.pow(2, -10 * progress);
      setDisplayValue(Math.floor(value * easeOutExpo));

      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      } else {
        setDisplayValue(value);
      }
    };

    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [value, duration]);

  return (
    <span className={className}>
      {prefix}
      {displayValue.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}
      {suffix}
    </span>
  );
};

/* ═══════════════════════════════════════════════════════════════════════════
   LOADING SKELETON
   Premium shimmer loading state
   ═══════════════════════════════════════════════════════════════════════════ */

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  width?: number | string;
  height?: number | string;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className,
  variant = 'text',
  width,
  height,
}) => {
  const variantStyles = {
    text: 'rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-lg',
  };

  return (
    <div
      className={cn(
        'animate-pulse bg-gradient-to-r from-white/5 via-white/10 to-white/5',
        'bg-[length:200%_100%] animate-shimmer',
        variantStyles[variant],
        className
      )}
      style={{ width, height: height || (variant === 'text' ? '1em' : undefined) }}
    />
  );
};

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION HEADER
   Premium section title with optional action
   ═══════════════════════════════════════════════════════════════════════════ */

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
  className?: string;
}

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  title,
  subtitle,
  action,
  className,
}) => {
  return (
    <div className={cn('flex items-center justify-between mb-6', className)}>
      <div>
        <h2 className="text-xl font-semibold text-white">{title}</h2>
        {subtitle && <p className="text-sm text-white/50 mt-1">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
};

export default {
  GlassPanel,
  LuxuryButton,
  MetricDisplay,
  StatusBadge,
  AnimatedCounter,
  Skeleton,
  SectionHeader,
};
