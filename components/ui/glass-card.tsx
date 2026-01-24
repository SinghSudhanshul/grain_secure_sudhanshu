/**
 * Glassmorphism Card Component
 * Apple-inspired design with backdrop blur and subtle shadows
 */

import React from 'react';
import { cn } from '@/lib/utils';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'default' | 'elevated' | 'subtle';
  hover?: boolean;
}

export function GlassCard({
  children,
  className,
  variant = 'default',
  hover = false
}: GlassCardProps) {
  const variants = {
    default: 'bg-white/80 backdrop-blur-md shadow-lg',
    elevated: 'bg-white/90 backdrop-blur-xl shadow-2xl',
    subtle: 'bg-white/60 backdrop-blur-sm shadow-md'
  };

  return (
    <div
      className={cn(
        'rounded-2xl border border-white/20 transition-all duration-300',
        variants[variant],
        hover && 'hover:shadow-xl hover:scale-[1.02] cursor-pointer',
        className
      )}
    >
      {children}
    </div>
  );
}

interface GlassCardHeaderProps {
  children: React.ReactNode;
  className?: string;
}

export function GlassCardHeader({ children, className }: GlassCardHeaderProps) {
  return (
    <div className={cn('p-6 border-b border-gray-200/50', className)}>
      {children}
    </div>
  );
}

interface GlassCardTitleProps {
  children: React.ReactNode;
  className?: string;
}

export function GlassCardTitle({ children, className }: GlassCardTitleProps) {
  return (
    <h3 className={cn('text-2xl font-semibold text-gray-900', className)}>
      {children}
    </h3>
  );
}

interface GlassCardDescriptionProps {
  children: React.ReactNode;
  className?: string;
}

export function GlassCardDescription({ children, className }: GlassCardDescriptionProps) {
  return (
    <p className={cn('text-sm text-gray-600 mt-1', className)}>
      {children}
    </p>
  );
}

interface GlassCardContentProps {
  children: React.ReactNode;
  className?: string;
}

export function GlassCardContent({ children, className }: GlassCardContentProps) {
  return (
    <div className={cn('p-6', className)}>
      {children}
    </div>
  );
}

interface GlassCardFooterProps {
  children: React.ReactNode;
  className?: string;
}

export function GlassCardFooter({ children, className }: GlassCardFooterProps) {
  return (
    <div className={cn('p-6 border-t border-gray-200/50 bg-gray-50/50', className)}>
      {children}
    </div>
  );
}
