'use client';

import { Suspense } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, formatValue, getCategoryColor, getCategoryLabel } from '@/lib/api';
import { useSearchParams, useRouter } from 'next/navigation';
import { useState, useMemo } from 'react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
  Line,
} from 'recharts';
import { TrendingUp, MapPin, Calendar } from 'lucide-react';

function ChartsContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const metricCode = searchParams.get('metric') || 'income_tax';
  const [areaCode, setAreaCode] = useState('UK');
  const [selectedSnapshots, setSelectedSnapshots] = useState<string[]>([]);

  const { data: metrics } = useQuery({
    queryKey: ['metrics'],
    queryFn: () => api.metrics(),
  });

  const { data: areas } = useQuery({
    queryKey: ['areas', 'flat'],
    queryFn: () => api.areas(true),
  });

  const { data: snapshots } = useQuery({
    queryKey: ['snapshots', metricCode],
    queryFn: () => api.snapshots(metricCode),
    enabled: !!metricCode,
  });

  const { data: metric } = useQuery({
    queryKey: ['metric', metricCode],
    queryFn: () => api.metric(metricCode),
    enabled: !!metricCode,
  });

  const snapshotsToFetch = selectedSnapshots.length > 0 ? selectedSnapshots : snapshots?.slice(0, 1).map(s => s.date) || [];

  const timeseriesQueries = useQuery({
    queryKey: ['timeseries-multi', metricCode, areaCode, snapshotsToFetch],
    queryFn: async () => {
      const results = await Promise.all(
        snapshotsToFetch.map(async (snapshotDate) => {
          const data = await api.timeseries(metricCode, areaCode, snapshotDate);
          return { snapshotDate, data };
        })
      );
      return results;
    },
    enabled: !!metricCode && snapshotsToFetch.length > 0,
  });

  const isLoading = timeseriesQueries.isLoading;

  const chartData = useMemo(() => {
    if (!timeseriesQueries.data) return [];

    const yearMap = new Map<number, any>();

    timeseriesQueries.data.forEach(({ snapshotDate, data }) => {
      data.forEach((point) => {
        if (!yearMap.has(point.year)) {
          yearMap.set(point.year, { year: point.year, isForecast: point.is_forecast });
        }
        yearMap.get(point.year)[snapshotDate] = point.value;
      });
    });

    return Array.from(yearMap.values()).sort((a, b) => a.year - b.year);
  }, [timeseriesQueries.data]);

  const snapshotColors = [
    '#d97706', // amber-600
    '#0ea5e9', // sky-500
    '#8b5cf6', // violet-500
    '#10b981', // emerald-500
    '#f59e0b', // amber-500
    '#ec4899', // pink-500
  ];

  const currentYear = new Date().getFullYear();
  const unit = metric?.unit || 'gbp';

  const formatYAxis = (value: number) => {
    if (unit === 'gbp') {
      if (Math.abs(value) >= 1e9) return `£${(value / 1e9).toFixed(0)}bn`;
      if (Math.abs(value) >= 1e6) return `£${(value / 1e6).toFixed(0)}m`;
      return `£${value}`;
    }
    if (unit === 'rate') return `${(value * 100).toFixed(0)}%`;
    if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(1)}m`;
    if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(0)}k`;
    return value.toString();
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    const data = payload[0].payload;
    return (
      <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3">
        <p className="text-sm font-medium text-slate-900 mb-2">
          {label}
          {data.isForecast && (
            <span className="ml-2 text-xs text-sky-600">(forecast)</span>
          )}
        </p>
        <div className="space-y-1">
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: entry.stroke || entry.color }}
              />
              <span className="text-xs text-slate-500">{entry.name}:</span>
              <span className="text-sm font-semibold text-slate-900">
                {formatValue(entry.value, unit)}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="p-8">
      {/* Controls */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-6 mb-6">
        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[250px]">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              <TrendingUp className="w-4 h-4 inline mr-1" />
              Metric
            </label>
            <select
              value={metricCode}
              onChange={(e) => {
                router.push(`/charts?metric=${e.target.value}`);
                setSelectedSnapshots([]);
              }}
              className="w-full px-3 py-2 rounded-lg border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-amber-500/40"
            >
              {metrics?.map((m) => (
                <option key={m.code} value={m.code}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              <MapPin className="w-4 h-4 inline mr-1" />
              Area
            </label>
            <select
              value={areaCode}
              onChange={(e) => setAreaCode(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-slate-300 bg-white focus:outline-none focus:ring-2 focus:ring-amber-500/40"
            >
              {areas?.map((a) => (
                <option key={a.code} value={a.code}>
                  {a.name}
                </option>
              ))}
            </select>
          </div>
        </div>
        {snapshots && snapshots.length > 1 && (
          <div className="mt-4">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              <Calendar className="w-4 h-4 inline mr-1" />
              Forecast snapshots (select multiple to compare)
            </label>
            <div className="flex flex-wrap gap-2">
              {snapshots.map((snapshot, idx) => {
                const isSelected = selectedSnapshots.includes(snapshot.date);
                const isDefault = selectedSnapshots.length === 0 && idx === 0;
                const color = snapshotColors[selectedSnapshots.indexOf(snapshot.date)] || snapshotColors[0];

                return (
                  <button
                    key={snapshot.date}
                    onClick={() => {
                      setSelectedSnapshots((prev) =>
                        prev.includes(snapshot.date)
                          ? prev.filter((s) => s !== snapshot.date)
                          : [...prev, snapshot.date]
                      );
                    }}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                      isSelected || isDefault
                        ? 'bg-slate-900 text-white'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                    }`}
                    style={
                      isSelected
                        ? {
                            backgroundColor: color,
                            color: 'white',
                          }
                        : {}
                    }
                  >
                    {snapshot.label}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Metric info */}
      {metric && (
        <div className="bg-white rounded-xl border border-slate-200/80 p-6 mb-6 animate-fade-in">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span
                  className="inline-flex items-center px-2.5 py-0.5 rounded text-xs font-medium"
                  style={{
                    backgroundColor: `${getCategoryColor(metric.category)}20`,
                    color: getCategoryColor(metric.category),
                  }}
                >
                  {getCategoryLabel(metric.category)}
                </span>
                <span className="text-xs text-slate-400 font-mono">{metric.unit}</span>
              </div>
              <h2 className="text-2xl font-editorial font-semibold text-slate-900">
                {metric.name}
              </h2>
              <p className="text-slate-500 font-mono text-sm mt-1">{metric.code}</p>
            </div>
            {chartData.length > 0 && snapshotsToFetch.length > 0 && (
              <div className="text-right">
                <p className="text-sm text-slate-500">Latest value ({snapshotsToFetch[0]})</p>
                <p className="text-2xl font-semibold text-slate-900 font-data">
                  {formatValue(chartData[chartData.length - 1]?.[snapshotsToFetch[0]] || 0, unit)}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-6 animate-fade-in">
        {isLoading ? (
          <div className="h-96 flex items-center justify-center">
            <div className="animate-spin w-8 h-8 border-2 border-slate-300 border-t-amber-500 rounded-full" />
          </div>
        ) : chartData.length === 0 ? (
          <div className="h-96 flex items-center justify-center text-slate-500">
            No data available for this metric and area
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="year" tick={{ fill: '#64748b', fontSize: 12 }} />
              <YAxis tickFormatter={formatYAxis} tick={{ fill: '#64748b', fontSize: 12 }} width={80} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine x={currentYear} stroke="#94a3b8" strokeDasharray="5 5" />
              {snapshotsToFetch.map((snapshotDate, idx) => (
                <Line
                  key={snapshotDate}
                  type="monotone"
                  dataKey={snapshotDate}
                  name={snapshotDate}
                  stroke={snapshotColors[idx % snapshotColors.length]}
                  strokeWidth={2.5}
                  dot={{ r: 4, fill: snapshotColors[idx % snapshotColors.length], stroke: 'white', strokeWidth: 2 }}
                  connectNulls
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        )}
        <div className="flex items-center justify-center flex-wrap gap-4 mt-4 pt-4 border-t border-slate-100">
          {snapshotsToFetch.map((snapshotDate, idx) => (
            <div key={snapshotDate} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: snapshotColors[idx % snapshotColors.length] }}
              />
              <span className="text-sm text-slate-600">{snapshotDate}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function ChartsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-cream-50 to-cream-100">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-8 py-6">
          <h1 className="text-3xl font-editorial font-semibold text-slate-900">Time series</h1>
          <p className="text-slate-600 mt-1">Historical data and forecasts</p>
        </div>
      </header>
      <Suspense fallback={<div className="p-8"><div className="animate-pulse bg-slate-200 h-96 rounded-xl" /></div>}>
        <ChartsContent />
      </Suspense>
    </div>
  );
}
