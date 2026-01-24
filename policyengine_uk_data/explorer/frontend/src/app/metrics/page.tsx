'use client';

import { Suspense } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, getCategoryColor, getCategoryLabel } from '@/lib/api';
import { useSearchParams, useRouter } from 'next/navigation';
import { Search, TrendingUp, ArrowRight } from 'lucide-react';
import Link from 'next/link';
import { useState, useMemo } from 'react';

function MetricsContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const categoryFilter = searchParams.get('category');
  const [search, setSearch] = useState('');

  const { data: metrics, isLoading } = useQuery({
    queryKey: ['metrics', categoryFilter],
    queryFn: () => api.metrics(categoryFilter || undefined),
  });

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: api.stats,
  });

  const categories = useMemo(() => {
    if (!stats?.categories) return [];
    return Object.entries(stats.categories).sort((a, b) => b[1] - a[1]);
  }, [stats]);

  const filteredMetrics = useMemo(() => {
    if (!metrics) return [];
    if (!search) return metrics;
    const lower = search.toLowerCase();
    return metrics.filter(
      (m) => m.code.toLowerCase().includes(lower) || m.name.toLowerCase().includes(lower)
    );
  }, [metrics, search]);

  return (
    <div className="p-8">
      <div className="flex gap-8">
        <div className="w-64 flex-shrink-0">
          <div className="bg-white rounded-xl border border-slate-200/80 p-4 sticky top-28">
            <h3 className="text-sm font-semibold text-slate-900 mb-3">Categories</h3>
            <div className="space-y-1">
              <button
                onClick={() => router.push('/metrics')}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                  !categoryFilter ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                All categories
              </button>
              {categories.map(([cat, count]) => (
                <button
                  key={cat}
                  onClick={() => router.push(`/metrics?category=${cat}`)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors flex items-center justify-between ${
                    categoryFilter === cat ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  <span>{getCategoryLabel(cat)}</span>
                  <span className="text-xs text-slate-400">{count}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="flex-1">
          <div className="relative mb-6">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              placeholder="Search metrics..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-12 pr-4 py-3 rounded-xl border border-slate-200 bg-white focus:outline-none focus:ring-2 focus:ring-amber-500/40"
            />
          </div>
          {isLoading ? (
            <div className="grid grid-cols-2 gap-4">
              {[...Array(8)].map((_, i) => (
                <div key={i} className="bg-white rounded-xl border border-slate-200 p-6 animate-pulse">
                  <div className="h-4 bg-slate-200 rounded w-3/4 mb-2" />
                  <div className="h-3 bg-slate-200 rounded w-1/2" />
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              {filteredMetrics.map((metric, i) => (
                <Link
                  key={metric.code}
                  href={`/charts?metric=${metric.code}`}
                  className="group bg-white rounded-xl border border-slate-200/80 p-5 hover:shadow-md transition-all animate-fade-in"
                  style={{ animationDelay: `${i * 20}ms` }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span
                          className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
                          style={{
                            backgroundColor: `${getCategoryColor(metric.category)}20`,
                            color: getCategoryColor(metric.category),
                          }}
                        >
                          {getCategoryLabel(metric.category)}
                        </span>
                        <span className="text-xs text-slate-400 font-mono">{metric.unit}</span>
                      </div>
                      <h3 className="font-medium text-slate-900 group-hover:text-amber-600 transition-colors truncate">
                        {metric.name}
                      </h3>
                      <p className="text-sm text-slate-500 font-mono">{metric.code}</p>
                    </div>
                    <ArrowRight className="w-5 h-5 text-slate-300 group-hover:text-amber-500 group-hover:translate-x-1 transition-all flex-shrink-0 mt-1" />
                  </div>
                </Link>
              ))}
            </div>
          )}
          {filteredMetrics.length === 0 && !isLoading && (
            <div className="text-center py-12">
              <TrendingUp className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-500">No metrics found</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function MetricsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-cream-50 to-cream-100">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-8 py-6">
          <h1 className="text-3xl font-editorial font-semibold text-slate-900">Metrics</h1>
          <p className="text-slate-600 mt-1">Browse available metrics</p>
        </div>
      </header>
      <Suspense fallback={<div className="p-8"><div className="animate-pulse bg-slate-200 h-96 rounded-xl" /></div>}>
        <MetricsContent />
      </Suspense>
    </div>
  );
}
