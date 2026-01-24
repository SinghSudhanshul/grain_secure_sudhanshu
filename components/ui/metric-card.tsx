/**
 * Animated Metric Card
 * Premium metric display with smooth animations and gradients
 */

'use client';

import React, { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon?: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  gradient?: 'blue' | 'purple' | 'green' | 'orange' | 'red';
  className?: string;
}

export function MetricCard({
  title,
  value,
  change,
  icon,
  trend = 'neutral',
  gradient = 'blue',
  className
}: MetricCardProps) {
  const [displayValue, setDisplayValue] = useState(0);

  const gradients = {
    blue: 'from-blue-500 to-blue-600',
    purple: 'from-purple-500 to-purple-600',
    green: 'from-green-500 to-green-600',
    orange: 'from-orange-500 to-orange-600',
    red: 'from-red-500 to-red-600'
  };

  const trendColors = {
    up: 'text-green-600',
    down: 'text-red-600',
    neutral: 'text-gray-600'
  };

  // Animate number counting
  useEffect(() => {
    if (typeof value === 'number') {
      let start = 0;
      const end = value;
      const duration = 1000;
      const increment = end / (duration / 16);

      const timer = setInterval(() => {
        start += increment;
        if (start >= end) {
          setDisplayValue(end);
          clearInterval(timer);
        } else {
          setDisplayValue(Math.floor(start));
        }
      }, 16);

      return () => clearInterval(timer);
    }
  }, [value]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      whileHover={{ scale: 1.02, y: -4 }}
      className={cn(
        'relative overflow-hidden rounded-2xl p-6',
        'bg-gradient-to-br shadow-lg',
        gradients[gradient],
        'transition-all duration-300',
        className
      )}
    >
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_120%,rgba(255,255,255,0.3),transparent)]" />
      </div>

      {/* Content */}
      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div className="text-white/80 text-sm font-medium">{title}</div>
          {icon && (
            <div className="text-white/60">
              {icon}
            </div>
          )}
        </div>

        <div className="text-white text-4xl font-bold mb-2">
          {typeof value === 'number' ? displayValue.toLocaleString() : value}
        </div>

        {change !== undefined && (
          <div className={cn('flex items-center text-sm font-medium', 'text-white/90')}>
            <span className={cn(
              'flex items-center gap-1',
              trend === 'up' && 'text-green-200',
              trend === 'down' && 'text-red-200'
            )}>
              {trend === 'up' && '↑'}
              {trend === 'down' && '↓'}
              {Math.abs(change)}%
            </span>
            <span className="ml-2 text-white/70">vs last period</span>
          </div>
        )}
      </div>

      {/* Shine effect */}
      <div className="absolute top-0 -left-full h-full w-1/2 bg-gradient-to-r from-transparent via-white/10 to-transparent skew-x-12 animate-shine" />
    </motion.div>
  );
}
