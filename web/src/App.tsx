import { useState, useEffect, useCallback } from 'react';
import type { AppConfig, Status } from './lib/api';
import * as api from './lib/api';
import { ProvidersPage } from './pages/ProvidersPage';
import { RoutesPage } from './pages/RoutesPage';
import { TagsPage } from './pages/TagsPage';
import { LogsPage } from './pages/LogsPage';
import { StatusDot } from './components/StatusDot';

type Tab = 'logs' | 'providers' | 'routes' | 'tags';

const tabs: { key: Tab; label: string; icon: string }[] = [
  { key: 'logs', label: 'Logs', icon: '📋' },
  { key: 'routes', label: 'Routes', icon: '🔀' },
  { key: 'providers', label: 'Providers', icon: '🔌' },
  { key: 'tags', label: 'Tags', icon: '🏷️' },
];

function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [tab, setTab] = useState<Tab>('logs');

  const refresh = useCallback(async () => {
    const [c, s] = await Promise.all([api.getConfig(), api.getStatus()]);
    setConfig(c);
    setStatus(s);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  if (!config || !status) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', color: 'var(--text-muted)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 20, height: 20, border: '2px solid var(--border)', borderTopColor: 'var(--accent)', borderRadius: '50%', animation: 'spin 0.8s linear infinite', display: 'inline-block' }} />
          Loading...
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
      </div>
    );
  }

  const handleConfigChange = (newConfig: AppConfig) => {
    setConfig(newConfig);
  };

  const handleTakeoverToggle = async () => {
    if (status.takeover.active) {
      await api.restoreClaude();
    } else {
      await api.takeoverClaude();
    }
    await refresh();
  };

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: '20px 24px', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 'var(--radius-sm)',
            background: 'linear-gradient(135deg, var(--accent), #8B5CF6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 16, fontWeight: 700, color: '#fff',
          }}>M</div>
          <div>
            <h1 style={{ margin: 0, fontSize: 18, fontWeight: 700, letterSpacing: '-0.3px' }}>Model Router</h1>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'monospace' }}>127.0.0.1:{config.port}</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <StatusDot active={status.takeover.active} />
          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            {status.takeover.active ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Takeover bar */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '10px 16px', marginBottom: 20,
        background: status.takeover.active ? 'var(--success-dim)' : 'var(--bg-card)',
        border: `1px solid ${status.takeover.active ? 'rgba(34,197,94,0.2)' : 'var(--border)'}`,
        borderRadius: 'var(--radius-md)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 13, color: status.takeover.active ? 'var(--success)' : 'var(--text-secondary)' }}>
            Claude Code Takeover
          </span>
        </div>
        {/* Toggle switch */}
        <div
          onClick={handleTakeoverToggle}
          style={{
            width: 40, height: 22, borderRadius: 11, cursor: 'pointer',
            background: status.takeover.active ? 'var(--success)' : '#333',
            position: 'relative', transition: 'background var(--transition)',
          }}
        >
          <div style={{
            width: 18, height: 18, borderRadius: 9,
            background: '#fff', position: 'absolute', top: 2,
            left: status.takeover.active ? 20 : 2,
            transition: 'left var(--transition)',
            boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
          }} />
        </div>
      </div>

      {/* Tabs */}
      <div style={{
        display: 'flex', gap: 0, marginBottom: 20,
        borderBottom: '1px solid var(--border)',
      }}>
        {tabs.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            style={{
              padding: '10px 20px', fontSize: 13, cursor: 'pointer',
              background: 'transparent', border: 'none',
              color: tab === t.key ? 'var(--text-primary)' : 'var(--text-muted)',
              borderBottom: tab === t.key ? '2px solid var(--accent)' : '2px solid transparent',
              fontWeight: tab === t.key ? 600 : 400,
              display: 'flex', alignItems: 'center', gap: 6,
            }}
          >
            <span style={{ fontSize: 14 }}>{t.icon}</span>
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div>
        {tab === 'logs' && <LogsPage />}
        {tab === 'providers' && <ProvidersPage config={config} onConfigChange={handleConfigChange} />}
        {tab === 'routes' && <RoutesPage config={config} onConfigChange={handleConfigChange} />}
        {tab === 'tags' && <TagsPage config={config} onConfigChange={handleConfigChange} />}
      </div>
    </div>
  );
}

export default App;
