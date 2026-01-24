'use client';

import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { Workflow, Box, ArrowRight, Layers } from 'lucide-react';

const groupColors: Record<string, string> = {
  raw_data: 'bg-slate-100 text-slate-700 border-slate-200',
  models: 'bg-violet-100 text-violet-700 border-violet-200',
  imputations: 'bg-sky-100 text-sky-700 border-sky-200',
  calibration: 'bg-emerald-100 text-emerald-700 border-emerald-200',
  outputs: 'bg-amber-100 text-amber-700 border-amber-200',
  targets: 'bg-pink-100 text-pink-700 border-pink-200',
};

export default function PipelinePage() {
  const { data: dagster, isLoading, error } = useQuery({
    queryKey: ['dagster'],
    queryFn: api.dagsterAssets,
  });

  const assetsByGroup = dagster?.assets.reduce(
    (acc, asset) => {
      const group = asset.group || 'ungrouped';
      if (!acc[group]) acc[group] = [];
      acc[group].push(asset);
      return acc;
    },
    {} as Record<string, typeof dagster.assets>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-cream-50 to-cream-100">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-8 py-6">
          <h1 className="text-3xl font-editorial font-semibold text-slate-900">
            Dagster pipeline
          </h1>
          <p className="text-slate-600 mt-1">
            Asset definitions and dependencies
          </p>
        </div>
      </header>

      <div className="p-8">
        {error || dagster?.error ? (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-700">
              Failed to load Dagster metadata: {dagster?.error || 'Connection error'}
            </p>
          </div>
        ) : isLoading ? (
          <div className="grid grid-cols-2 gap-6">
            {[...Array(4)].map((_, i) => (
              <div
                key={i}
                className="bg-white rounded-xl border border-slate-200 p-6 animate-pulse"
              >
                <div className="h-6 bg-slate-200 rounded w-32 mb-4" />
                <div className="space-y-3">
                  <div className="h-4 bg-slate-200 rounded w-full" />
                  <div className="h-4 bg-slate-200 rounded w-3/4" />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-6">
            {/* Summary */}
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-white rounded-xl border border-slate-200/80 p-5">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-slate-100 text-slate-600">
                    <Box className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-2xl font-semibold text-slate-900">
                      {dagster?.assets.length || 0}
                    </p>
                    <p className="text-sm text-slate-500">Total assets</p>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-xl border border-slate-200/80 p-5">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-violet-100 text-violet-600">
                    <Layers className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-2xl font-semibold text-slate-900">
                      {dagster?.groups.length || 0}
                    </p>
                    <p className="text-sm text-slate-500">Groups</p>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-xl border border-slate-200/80 p-5 col-span-2">
                <p className="text-sm text-slate-500 mb-2">Groups</p>
                <div className="flex flex-wrap gap-2">
                  {dagster?.groups.map((g) => (
                    <span
                      key={g}
                      className={`px-2.5 py-1 rounded-full text-xs font-medium border ${
                        groupColors[g] || 'bg-slate-100 text-slate-700 border-slate-200'
                      }`}
                    >
                      {g}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Assets by group */}
            {assetsByGroup &&
              Object.entries(assetsByGroup).map(([group, assets], groupIdx) => (
                <div
                  key={group}
                  className="bg-white rounded-xl border border-slate-200/80 overflow-hidden animate-fade-in"
                  style={{ animationDelay: `${groupIdx * 50}ms` }}
                >
                  <div
                    className={`px-6 py-4 border-b border-slate-200 flex items-center gap-3 ${
                      groupColors[group]?.replace('text-', 'bg-').split(' ')[0] ||
                      'bg-slate-50'
                    }`}
                  >
                    <Workflow className="w-5 h-5" />
                    <h2 className="font-semibold text-slate-900 capitalize">
                      {group.replace(/_/g, ' ')}
                    </h2>
                    <span className="text-sm text-slate-500">
                      {assets.length} asset{assets.length !== 1 ? 's' : ''}
                    </span>
                  </div>

                  <div className="divide-y divide-slate-100">
                    {assets.map((asset, i) => (
                      <div
                        key={asset.key}
                        className="px-6 py-4 hover:bg-slate-50 transition-colors"
                      >
                        <div className="flex items-start justify-between">
                          <div>
                            <h3 className="font-medium text-slate-900 font-mono">
                              {asset.key}
                            </h3>
                            {asset.description && (
                              <p className="text-sm text-slate-500 mt-1">
                                {asset.description}
                              </p>
                            )}
                          </div>
                        </div>

                        {asset.deps.length > 0 && (
                          <div className="mt-3 flex items-center gap-2 flex-wrap">
                            <span className="text-xs text-slate-400">Dependencies:</span>
                            {asset.deps.map((dep) => (
                              <span
                                key={dep}
                                className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-100 text-xs text-slate-600 font-mono"
                              >
                                <ArrowRight className="w-3 h-3" />
                                {dep}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}
