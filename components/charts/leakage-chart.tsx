/**
 * Enhanced Leakage Chart Component
 * Visualizes PDS leakage trends over time with comparison
 */

'use client';

import React from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { GlassCard, GlassCardHeader, GlassCardTitle, GlassCardContent } from '@/components/ui/glass-card';

interface LeakageDataPoint {
  date: string;
  leakagePercentage: number;
  estimatedLoss: number;
  district?: string;
}

interface LeakageChartProps {
  data: LeakageDataPoint[];
  comparisonData?: LeakageDataPoint[];
  title?: string;
  showLoss?: boolean;
}

export function LeakageChart({
  data,
  comparisonData,
  title = 'Leakage Trend Analysis',
  showLoss = false
}: LeakageChartProps) {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white/95 backdrop-blur-md p-4 rounded-lg shadow-xl border border-gray-200">
          <p className="font-semibold text-gray-900 mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center gap-2 text-sm">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-gray-600">{entry.name}:</span>
              <span className="font-semibold text-gray-900">
                {entry.value.toFixed(2)}%
              </span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <GlassCard>
      <GlassCardHeader>
        <GlassCardTitle>{title}</GlassCardTitle>
      </GlassCardHeader>
      <GlassCardContent>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="leakageGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
              {comparisonData && (
                <linearGradient id="comparisonGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              )}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="date"
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
            />
            <YAxis
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
              label={{ value: 'Leakage %', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ paddingTop: '20px' }}
              iconType="circle"
            />
            <Area
              type="monotone"
              dataKey="leakagePercentage"
              stroke="#ef4444"
              strokeWidth={3}
              fill="url(#leakageGradient)"
              name="Current District"
            />
            {comparisonData && (
              <Area
                type="monotone"
                data={comparisonData}
                dataKey="leakagePercentage"
                stroke="#3b82f6"
                strokeWidth={2}
                fill="url(#comparisonGradient)"
                name="Comparison District"
              />
            )}
          </AreaChart>
        </ResponsiveContainer>

        {showLoss && (
          <div className="mt-6 grid grid-cols-3 gap-4">
            <div className="text-center p-4 bg-red-50 rounded-lg">
              <div className="text-sm text-red-600 font-medium">Total Leakage</div>
              <div className="text-2xl font-bold text-red-700 mt-1">
                {data.reduce((sum, d) => sum + d.leakagePercentage, 0) / data.length}%
              </div>
            </div>
            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <div className="text-sm text-orange-600 font-medium">Estimated Loss</div>
              <div className="text-2xl font-bold text-orange-700 mt-1">
                ₹{(data.reduce((sum, d) => sum + d.estimatedLoss, 0) / 1000000).toFixed(2)}M
              </div>
            </div>
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-sm text-blue-600 font-medium">Trend</div>
              <div className="text-2xl font-bold text-blue-700 mt-1">
                {data[data.length - 1].leakagePercentage < data[0].leakagePercentage ? '↓' : '↑'}
                {Math.abs(data[data.length - 1].leakagePercentage - data[0].leakagePercentage).toFixed(1)}%
              </div>
            </div>
          </div>
        )}
      </GlassCardContent>
    </GlassCard>
  );
}
