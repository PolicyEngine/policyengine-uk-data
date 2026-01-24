'use client';

import { useQuery } from '@tanstack/react-query';
import { api, getCategoryColor, getCategoryLabel, formatValue } from '@/lib/api';
import {
  BarChart3,
  Calendar,
  Database,
  Globe2,
  TrendingUp,
  ArrowRight,
} from 'lucide-react';
import Link from 'next/link';

function StatCard({
  label,
  value,
  icon: Icon,
  color = 'slate',
  delay = 0,
}: {
  label: string;
  value: string | number;
  icon: React.ElementType;
  color?: string;
  delay?: number;
}) {
  const colorClasses: Record<string, string> = {
    slate: 'bg-slate-100 text-slate-600',
    amber: 'bg-amber-100 text-amber-600',
    violet: 'bg-violet-100 text-violet-600',
    emerald: 'bg-emerald-100 text-emerald-600',
  };

  return (
    <div
      className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm hover:shadow-md transition-shadow animate-fade-in"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-500 mb-1">{label}</p>
          <p className="text-3xl font-editorial font-semibold text-slate-900">
            {typeof value === 'number' ? value.toLocaleString() : value}
          </p>
        </div>
        <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </div>
  );
}

function CategoryBreakdown({ categories }: { categories: Record<string, number> }) {
  const total = Object.values(categories).reduce((a, b) => a + b, 0);

  return (
    <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm animate-fade-in" style={{ animationDelay: '200ms' }}>
      <h3 className="text-lg font-editorial font-semibold text-slate-900 mb-4">
        Metrics by category
      </h3>
      <div className="space-y-3">
        {Object.entries(categories)
          .sort((a, b) => b[1] - a[1])
          .map(([category, count]) => {
            const percentage = (count / total) * 100;
            return (
              <div key={category}>
                <div className="flex items-center justify-between text-sm mb-1">
                  <span
                    className="font-medium"
                    style={{ color: getCategoryColor(category) }}
                  >
                    {getCategoryLabel(category)}
                  </span>
                  <span className="text-slate-500">{count} metrics</span>
                </div>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${percentage}%`,
                      backgroundColor: getCategoryColor(category),
                    }}
                  />
                </div>
              </div>
            );
          })}
      </div>
    </div>
  );
}

function SourcesList({ sources }: { sources: Record<string, number> }) {
  const sortedSources = Object.entries(sources)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);

  return (
    <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm animate-fade-in" style={{ animationDelay: '250ms' }}>
      <h3 className="text-lg font-editorial font-semibold text-slate-900 mb-4">
        Data sources
      </h3>
      <div className="space-y-2">
        {sortedSources.map(([source, count]) => (
          <div
            key={source}
            className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0"
          >
            <span className="text-sm text-slate-700 truncate max-w-[200px]">
              {source}
            </span>
            <span className="text-sm font-medium text-slate-500 font-data">
              {count.toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function QuickLinks() {
  const links = [
    {
      href: '/metrics?category=obr',
      label: 'OBR forecasts',
      description: 'Tax receipts and welfare spending',
      color: 'violet',
    },
    {
      href: '/charts?metric=income_tax',
      label: 'Income tax time series',
      description: 'Historical and forecast data',
      color: 'amber',
    },
    {
      href: '/areas',
      label: 'Geographic breakdown',
      description: 'UK, countries, and regions',
      color: 'emerald',
    },
    {
      href: '/pipeline',
      label: 'Dagster pipeline',
      description: 'Asset status and dependencies',
      color: 'slate',
    },
  ];

  return (
    <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm animate-fade-in" style={{ animationDelay: '300ms' }}>
      <h3 className="text-lg font-editorial font-semibold text-slate-900 mb-4">
        Quick links
      </h3>
      <div className="space-y-2">
        {links.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className="group flex items-center justify-between p-3 rounded-lg hover:bg-slate-50 transition-colors"
          >
            <div>
              <p className="text-sm font-medium text-slate-900 group-hover:text-amber-600 transition-colors">
                {link.label}
              </p>
              <p className="text-xs text-slate-500">{link.description}</p>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-amber-500 group-hover:translate-x-1 transition-all" />
          </Link>
        ))}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { data: stats, isLoading, error } = useQuery({
    queryKey: ['stats'],
    queryFn: api.stats,
  });

  if (error) {
    return (
      <div className="p-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-700">
            Failed to connect to API. Make sure the backend is running on port 8000.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-cream-50 to-cream-100">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-8 py-6">
          <h1 className="text-3xl font-editorial font-semibold text-slate-900">
            Targets Explorer
          </h1>
          <p className="text-slate-600 mt-1">
            Calibration targets and official statistics for UK tax-benefit modelling
          </p>
        </div>
      </header>

      {/* Main content */}
      <div className="p-8">
        {isLoading ? (
          <div className="grid grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div
                key={i}
                className="bg-white rounded-xl border border-slate-200 p-6 h-32 animate-pulse"
              >
                <div className="h-4 bg-slate-200 rounded w-24 mb-3" />
                <div className="h-8 bg-slate-200 rounded w-16" />
              </div>
            ))}
          </div>
        ) : stats ? (
          <div className="space-y-6">
            {/* Stats row */}
            <div className="grid grid-cols-4 gap-6">
              <StatCard
                label="Observations"
                value={stats.total_observations}
                icon={Database}
                color="amber"
                delay={0}
              />
              <StatCard
                label="Metrics"
                value={stats.total_metrics}
                icon={TrendingUp}
                color="violet"
                delay={50}
              />
              <StatCard
                label="Areas"
                value={stats.total_areas}
                icon={Globe2}
                color="emerald"
                delay={100}
              />
              <StatCard
                label="Year range"
                value={`${stats.year_range[0]}–${stats.year_range[1]}`}
                icon={Calendar}
                color="slate"
                delay={150}
              />
            </div>

            {/* Detail cards */}
            <div className="grid grid-cols-3 gap-6">
              <CategoryBreakdown categories={stats.categories} />
              <SourcesList sources={stats.sources} />
              <QuickLinks />
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
