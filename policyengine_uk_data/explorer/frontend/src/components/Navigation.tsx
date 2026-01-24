'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  BarChart3,
  Database,
  Globe2,
  Home,
  Layers,
  TrendingUp,
  Workflow,
} from 'lucide-react';

const navItems = [
  { href: '/', label: 'Dashboard', icon: Home },
  { href: '/metrics', label: 'Metrics', icon: TrendingUp },
  { href: '/areas', label: 'Areas', icon: Globe2 },
  { href: '/observations', label: 'Observations', icon: Database },
  { href: '/charts', label: 'Time Series', icon: BarChart3 },
  { href: '/pipeline', label: 'Pipeline', icon: Workflow },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="fixed left-0 top-0 bottom-0 w-64 bg-slate-900 text-white flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center">
            <Layers className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-editorial text-lg font-semibold tracking-tight">
              Targets
            </h1>
            <p className="text-xs text-slate-400">PolicyEngine UK</p>
          </div>
        </div>
      </div>

      {/* Navigation Items */}
      <div className="flex-1 py-6 px-3 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`
                flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
                transition-all duration-150 ease-out
                ${
                  isActive
                    ? 'bg-white/10 text-white'
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }
              `}
            >
              <Icon className="w-5 h-5" />
              {item.label}
              {isActive && (
                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-amber-400" />
              )}
            </Link>
          );
        })}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-800">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>Connected to API</span>
        </div>
      </div>
    </nav>
  );
}
