const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Types
export interface Area {
  code: string;
  name: string;
  area_type: string;
  parent_code: string | null;
  children: Area[];
}

export interface Metric {
  code: string;
  name: string;
  category: string;
  unit: string;
}

export interface Observation {
  id: number;
  metric_code: string;
  area_code: string;
  valid_year: number;
  snapshot_date: string;
  value: number;
  source: string;
  source_url: string | null;
  is_forecast: boolean;
}

export interface TimeSeriesPoint {
  year: number;
  value: number;
  is_forecast: boolean;
  snapshot_date: string;
}

export interface Stats {
  total_observations: number;
  total_metrics: number;
  total_areas: number;
  year_range: [number, number];
  categories: Record<string, number>;
  sources: Record<string, number>;
}

export interface DagsterAsset {
  key: string;
  group: string | null;
  description: string | null;
  deps: string[];
}

export interface DagsterInfo {
  assets: DagsterAsset[];
  groups: string[];
  error?: string;
}

export interface Snapshot {
  date: string;
  label: string;
}

// API functions
async function fetchApi<T>(endpoint: string): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  // Health & stats
  health: () => fetchApi<{ status: string; observations: number }>('/api/health'),
  stats: () => fetchApi<Stats>('/api/stats'),

  // Areas
  areas: (flat = false) => fetchApi<Area[]>(`/api/areas?flat=${flat}`),
  area: (code: string) => fetchApi<Area>(`/api/areas/${code}`),

  // Metrics
  metrics: (category?: string) =>
    fetchApi<Metric[]>(`/api/metrics${category ? `?category=${category}` : ''}`),
  metric: (code: string) => fetchApi<Metric>(`/api/metrics/${code}`),

  // Observations
  observations: (params: {
    metric_code?: string;
    area_code?: string;
    year_from?: number;
    year_to?: number;
    source?: string;
    is_forecast?: boolean;
    limit?: number;
    offset?: number;
  }) => {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) searchParams.set(key, String(value));
    });
    return fetchApi<Observation[]>(`/api/observations?${searchParams}`);
  },

  timeseries: (metric_code: string, area_code = 'UK', snapshot_date?: string) => {
    const params = new URLSearchParams({ metric_code, area_code });
    if (snapshot_date) params.set('snapshot_date', snapshot_date);
    return fetchApi<TimeSeriesPoint[]>(`/api/observations/timeseries?${params}`);
  },

  snapshots: (metric_code?: string) =>
    fetchApi<Snapshot[]>(`/api/observations/snapshots${metric_code ? `?metric_code=${metric_code}` : ''}`),

  // Dagster
  dagsterAssets: () => fetchApi<DagsterInfo>('/api/dagster/assets'),
};

// Formatting utilities
export function formatValue(value: number, unit: string): string {
  if (unit === 'gbp') {
    if (Math.abs(value) >= 1e9) {
      return `£${(value / 1e9).toFixed(1)}bn`;
    }
    if (Math.abs(value) >= 1e6) {
      return `£${(value / 1e6).toFixed(1)}m`;
    }
    return `£${value.toLocaleString()}`;
  }
  if (unit === 'rate') {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (unit === 'count') {
    if (Math.abs(value) >= 1e6) {
      return `${(value / 1e6).toFixed(2)}m`;
    }
    if (Math.abs(value) >= 1e3) {
      return `${(value / 1e3).toFixed(1)}k`;
    }
    return value.toLocaleString();
  }
  return value.toLocaleString();
}

export function getCategoryColor(category: string): string {
  const colors: Record<string, string> = {
    obr: '#8b5cf6',
    dwp: '#f59e0b',
    ons: '#ec4899',
    hmrc: '#10b981',
    nts: '#06b6d4',
    voa: '#6366f1',
    sss: '#0ea5e9',
    housing: '#84cc16',
  };
  return colors[category] || '#64748b';
}

export function getCategoryLabel(category: string): string {
  const labels: Record<string, string> = {
    obr: 'OBR',
    dwp: 'DWP',
    ons: 'ONS',
    hmrc: 'HMRC',
    nts: 'NTS',
    voa: 'VOA',
    sss: 'Scottish Social Security',
    housing: 'Housing',
  };
  return labels[category] || category.toUpperCase();
}
