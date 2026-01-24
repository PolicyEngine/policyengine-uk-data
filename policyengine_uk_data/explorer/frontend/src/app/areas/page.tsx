'use client';

import { useQuery } from '@tanstack/react-query';
import { api, Area } from '@/lib/api';
import { ChevronRight, Globe2, MapPin, Building2 } from 'lucide-react';
import { useState } from 'react';
import Link from 'next/link';

function AreaTypeIcon({ type }: { type: string }) {
  switch (type) {
    case 'uk':
      return <Globe2 className="w-5 h-5" />;
    case 'country':
      return <MapPin className="w-5 h-5" />;
    case 'region':
      return <Building2 className="w-5 h-5" />;
    default:
      return <MapPin className="w-5 h-5" />;
  }
}

function AreaCard({ area, depth = 0 }: { area: Area; depth?: number }) {
  const [expanded, setExpanded] = useState(depth < 2);
  const hasChildren = area.children && area.children.length > 0;

  const typeColors: Record<string, string> = {
    uk: 'bg-amber-100 text-amber-700 border-amber-200',
    country: 'bg-violet-100 text-violet-700 border-violet-200',
    region: 'bg-emerald-100 text-emerald-700 border-emerald-200',
  };

  return (
    <div className="animate-fade-in" style={{ animationDelay: `${depth * 50}ms` }}>
      <div
        className={`
          flex items-center gap-3 p-4 rounded-lg border bg-white
          ${hasChildren ? 'cursor-pointer hover:bg-slate-50' : ''}
          transition-colors
        `}
        onClick={() => hasChildren && setExpanded(!expanded)}
      >
        {hasChildren && (
          <ChevronRight
            className={`w-4 h-4 text-slate-400 transition-transform ${
              expanded ? 'rotate-90' : ''
            }`}
          />
        )}
        {!hasChildren && <div className="w-4" />}

        <div
          className={`p-2 rounded-lg border ${typeColors[area.area_type] || 'bg-slate-100 text-slate-700 border-slate-200'}`}
        >
          <AreaTypeIcon type={area.area_type} />
        </div>

        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-slate-900">{area.name}</h3>
          <p className="text-sm text-slate-500 font-mono">{area.code}</p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 capitalize">{area.area_type}</span>
          <Link
            href={`/observations?area_code=${area.code}`}
            onClick={(e) => e.stopPropagation()}
            className="text-xs text-amber-600 hover:text-amber-700 font-medium"
          >
            View data →
          </Link>
        </div>
      </div>

      {hasChildren && expanded && (
        <div className="ml-8 mt-2 space-y-2 border-l-2 border-slate-100 pl-4">
          {area.children.map((child) => (
            <AreaCard key={child.code} area={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function AreasPage() {
  const { data: areas, isLoading } = useQuery({
    queryKey: ['areas'],
    queryFn: () => api.areas(false),
  });

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: api.stats,
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-cream-50 to-cream-100">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-8 py-6">
          <h1 className="text-3xl font-editorial font-semibold text-slate-900">
            Geographic areas
          </h1>
          <p className="text-slate-600 mt-1">
            {stats?.total_areas || 0} areas in the hierarchy
          </p>
        </div>
      </header>

      <div className="p-8">
        {/* Legend */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-4 mb-6 flex items-center gap-6">
          <span className="text-sm text-slate-500">Area types:</span>
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded bg-amber-100 text-amber-700">
              <Globe2 className="w-4 h-4" />
            </div>
            <span className="text-sm text-slate-700">UK</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded bg-violet-100 text-violet-700">
              <MapPin className="w-4 h-4" />
            </div>
            <span className="text-sm text-slate-700">Country</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded bg-emerald-100 text-emerald-700">
              <Building2 className="w-4 h-4" />
            </div>
            <span className="text-sm text-slate-700">Region</span>
          </div>
        </div>

        {/* Area tree */}
        {isLoading ? (
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div
                key={i}
                className="bg-white rounded-lg border border-slate-200 p-4 animate-pulse"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-slate-200 rounded-lg" />
                  <div className="flex-1">
                    <div className="h-4 bg-slate-200 rounded w-32 mb-2" />
                    <div className="h-3 bg-slate-200 rounded w-16" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-3">
            {areas?.map((area) => (
              <AreaCard key={area.code} area={area} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
