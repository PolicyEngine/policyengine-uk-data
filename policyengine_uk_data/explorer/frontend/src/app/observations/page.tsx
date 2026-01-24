'use client';

import { Suspense } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, formatValue, getCategoryColor } from '@/lib/api';
import { useSearchParams } from 'next/navigation';
import { useState } from 'react';
import { Database, Filter, ExternalLink } from 'lucide-react';

function ObservationsContent() {
  const searchParams = useSearchParams();

  const [filters, setFilters] = useState({
    metric_code: searchParams.get('metric_code') || '',
    area_code: searchParams.get('area_code') || '',
    year_from: searchParams.get('year_from') || '',
    year_to: searchParams.get('year_to') || '',
    source: searchParams.get('source') || '',
  });

  const { data: metrics } = useQuery({
    queryKey: ['metrics'],
    queryFn: () => api.metrics(),
  });

  const { data: areas } = useQuery({
    queryKey: ['areas', 'flat'],
    queryFn: () => api.areas(true),
  });

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: api.stats,
  });

  const { data: observations, isLoading } = useQuery({
    queryKey: ['observations', filters],
    queryFn: () =>
      api.observations({
        metric_code: filters.metric_code || undefined,
        area_code: filters.area_code || undefined,
        year_from: filters.year_from ? parseInt(filters.year_from) : undefined,
        year_to: filters.year_to ? parseInt(filters.year_to) : undefined,
        source: filters.source || undefined,
        limit: 100,
      }),
  });

  const metricMap = new Map(metrics?.map((m) => [m.code, m]) || []);
  const sources = stats?.sources ? Object.keys(stats.sources).sort() : [];

  const updateFilter = (key: string, value: string) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="p-8">
      <div className="bg-white rounded-xl border border-slate-200/80 p-6 mb-6">
        <div className="flex items-center gap-2 mb-4">
          <Filter className="w-5 h-5 text-slate-500" />
          <h2 className="font-medium text-slate-900">Filters</h2>
        </div>
        <div className="grid grid-cols-5 gap-4">
          <div>
            <label className="block text-sm text-slate-600 mb-1">Metric</label>
            <select
              value={filters.metric_code}
              onChange={(e) => updateFilter('metric_code', e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-slate-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/40"
            >
              <option value="">All metrics</option>
              {metrics?.map((m) => (
                <option key={m.code} value={m.code}>{m.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-slate-600 mb-1">Area</label>
            <select
              value={filters.area_code}
              onChange={(e) => updateFilter('area_code', e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-slate-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/40"
            >
              <option value="">All areas</option>
              {areas?.map((a) => (
                <option key={a.code} value={a.code}>{a.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-slate-600 mb-1">Year from</label>
            <input
              type="number"
              value={filters.year_from}
              onChange={(e) => updateFilter('year_from', e.target.value)}
              placeholder="2020"
              className="w-full px-3 py-2 rounded-lg border border-slate-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/40"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-600 mb-1">Year to</label>
            <input
              type="number"
              value={filters.year_to}
              onChange={(e) => updateFilter('year_to', e.target.value)}
              placeholder="2030"
              className="w-full px-3 py-2 rounded-lg border border-slate-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/40"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-600 mb-1">Source</label>
            <select
              value={filters.source}
              onChange={(e) => updateFilter('source', e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-slate-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/40"
            >
              <option value="">All sources</option>
              {sources.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200/80 overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-slate-500" />
            <span className="font-medium text-slate-900">{observations?.length || 0} results</span>
            {observations && observations.length >= 100 && (
              <span className="text-sm text-slate-500">(showing first 100)</span>
            )}
          </div>
        </div>
        {isLoading ? (
          <div className="p-8 text-center">
            <div className="animate-spin w-8 h-8 border-2 border-slate-300 border-t-amber-500 rounded-full mx-auto" />
          </div>
        ) : observations && observations.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50">
                  <th className="text-left font-medium text-slate-600 px-6 py-3">Metric</th>
                  <th className="text-left font-medium text-slate-600 px-6 py-3">Area</th>
                  <th className="text-left font-medium text-slate-600 px-6 py-3">Year</th>
                  <th className="text-right font-medium text-slate-600 px-6 py-3">Value</th>
                  <th className="text-left font-medium text-slate-600 px-6 py-3">Type</th>
                  <th className="text-left font-medium text-slate-600 px-6 py-3">Source</th>
                </tr>
              </thead>
              <tbody>
                {observations.map((obs, i) => {
                  const metric = metricMap.get(obs.metric_code);
                  return (
                    <tr key={obs.id} className="border-t border-slate-100 hover:bg-cream-50 transition-colors">
                      <td className="px-6 py-3">
                        <div className="flex items-center gap-2">
                          {metric && (
                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: getCategoryColor(metric.category) }} />
                          )}
                          <span className="font-medium text-slate-900">{metric?.name || obs.metric_code}</span>
                        </div>
                      </td>
                      <td className="px-6 py-3 text-slate-600">{obs.area_code}</td>
                      <td className="px-6 py-3 font-mono text-slate-600">{obs.valid_year}</td>
                      <td className="px-6 py-3 text-right font-mono font-medium text-slate-900">
                        {formatValue(obs.value, metric?.unit || 'count')}
                      </td>
                      <td className="px-6 py-3">
                        {obs.is_forecast ? (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-sky-100 text-sky-700">Forecast</span>
                        ) : (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-emerald-100 text-emerald-700">Actual</span>
                        )}
                      </td>
                      <td className="px-6 py-3">
                        {obs.source_url ? (
                          <a href={obs.source_url} target="_blank" rel="noopener noreferrer" className="text-amber-600 hover:text-amber-700 flex items-center gap-1">
                            {obs.source}<ExternalLink className="w-3 h-3" />
                          </a>
                        ) : (
                          <span className="text-slate-500">{obs.source}</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-12 text-center">
            <Database className="w-12 h-12 text-slate-300 mx-auto mb-4" />
            <p className="text-slate-500">No observations found</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ObservationsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-cream-50 to-cream-100">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-8 py-6">
          <h1 className="text-3xl font-editorial font-semibold text-slate-900">Observations</h1>
          <p className="text-slate-600 mt-1">Browse and filter data points</p>
        </div>
      </header>
      <Suspense fallback={<div className="p-8"><div className="animate-pulse bg-slate-200 h-96 rounded-xl" /></div>}>
        <ObservationsContent />
      </Suspense>
    </div>
  );
}
