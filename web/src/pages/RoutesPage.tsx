import { useState } from 'react';
import type { AppConfig, Route, TestResult } from '../lib/api';
import * as api from '../lib/api';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Badge } from '../components/Badge';
import { Input, Select } from '../components/Input';

export function RoutesPage({ config, onConfigChange }: { config: AppConfig; onConfigChange: (c: AppConfig) => void }) {
  const [editing, setEditing] = useState<number | null>(null);
  const [adding, setAdding] = useState(false);
  const [testResults, setTestResults] = useState<Record<string, TestResult>>({});
  const [testing, setTesting] = useState<string | null>(null);

  const handleDelete = async (index: number) => {
    const newRoutes = config.routes.filter((_, i) => i !== index);
    const newConfig = { ...config, routes: newRoutes };
    await api.updateConfig(newConfig);
    onConfigChange(newConfig);
  };

  const handleTest = async (tag: string) => {
    setTesting(tag);
    try {
      const result = await api.testRoute(tag);
      setTestResults(prev => ({ ...prev, [tag]: result }));
    } catch (e: any) {
      setTestResults(prev => ({ ...prev, [tag]: { success: false, tag, provider: '', model: '', format: '', latency_ms: 0, error: e.message, response: null } }));
    }
    setTesting(null);
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Routes</h2>
        <Button variant="primary" onClick={() => setAdding(true)}>+ Add</Button>
      </div>

      {(adding || editing !== null) && (
        <RouteForm
          initial={editing !== null ? config.routes[editing] : undefined}
          providers={Object.keys(config.providers)}
          tags={config.tags.map(t => t.name)}
          onSave={async (route) => {
            const newRoutes = editing !== null
              ? config.routes.map((r, i) => i === editing ? route : r)
              : [...config.routes, route];
            const newConfig = { ...config, routes: newRoutes };
            await api.updateConfig(newConfig);
            onConfigChange(newConfig);
            setAdding(false);
            setEditing(null);
          }}
          onCancel={() => { setAdding(false); setEditing(null); }}
        />
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {config.routes.map((route, i) => {
          const tag = route.tags[0] || '';
          const tr = testResults[tag];
          return (
            <Card key={i}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 15, fontWeight: 600 }}>{route.model}</span>
                    <Badge color={route.format === 'openai' ? '#10a37f' : '#d97706'}>{route.format || 'openai'}</Badge>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                    via <span style={{ color: 'var(--text-secondary)' }}>{route.provider}</span>
                    <span className="mono" style={{ marginLeft: 8 }}>{route.endpoint}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
                    {route.tags.map(t => {
                      const tagDef = config.tags.find(td => td.name === t);
                      return <Badge key={t} color={tagDef?.color || '#555'}>{t}</Badge>;
                    })}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                  <Button
                    variant={testing === tag ? 'secondary' : 'success'}
                    onClick={() => handleTest(tag)}
                    disabled={!!testing}
                    style={{ fontSize: 12, padding: '4px 12px' }}
                  >
                    {testing === tag ? '⏳ Testing...' : '▶ Test'}
                  </Button>
                  <Button variant="ghost" onClick={() => setEditing(i)} style={{ fontSize: 12, padding: '4px 10px' }}>Edit</Button>
                  <Button variant="danger" onClick={() => handleDelete(i)} style={{ fontSize: 12, padding: '4px 10px' }}>Delete</Button>
                </div>
              </div>

              {tr && (
                <div style={{
                  marginTop: 12, padding: 10, borderRadius: 'var(--radius-sm)', fontSize: 12,
                  background: tr.success ? 'var(--success-dim)' : 'var(--danger-dim)',
                  border: `1px solid ${tr.success ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)'}`,
                }}>
                  {tr.success ? (
                    <div>
                      <span style={{ color: 'var(--success)', fontWeight: 600 }}>✓ OK</span>
                      <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>{tr.provider} / {tr.model}</span>
                      <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>{tr.latency_ms}ms</span>
                      {tr.response && (
                        <pre className="mono" style={{ marginTop: 6, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap', maxHeight: 100, overflow: 'auto', fontSize: 11, background: 'rgba(0,0,0,0.2)', padding: 8, borderRadius: 'var(--radius-sm)' }}>
                          {(() => {
                            const content = tr.response?.content;
                            if (Array.isArray(content)) {
                              const textBlock = content.find((b: any) => b.type === 'text');
                              if (textBlock) return textBlock.text;
                              return JSON.stringify(content, null, 2);
                            }
                            return JSON.stringify(tr.response, null, 2);
                          })()}
                        </pre>
                      )}
                    </div>
                  ) : (
                    <div style={{ color: 'var(--danger)' }}>✗ {tr.error}</div>
                  )}
                </div>
              )}
            </Card>
          );
        })}
        {config.routes.length === 0 && (
          <div style={{ padding: 48, textAlign: 'center', color: 'var(--text-muted)', border: '1px dashed var(--border)', borderRadius: 'var(--radius-md)' }}>
            <div style={{ fontSize: 28, marginBottom: 8 }}>🔀</div>
            No routes configured
          </div>
        )}
      </div>

      <Card style={{ marginTop: 20, background: 'rgba(0,0,0,0.2)' }}>
        <h3 style={{ margin: '0 0 8px 0', fontSize: 12, color: 'var(--text-muted)', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.5px' }}>curl test commands</h3>
        <pre className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)', whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>
{`# Test a specific route
curl -s http://127.0.0.1:${config.port}/api/test \\
  -H 'Content-Type: application/json' \\
  -d '{"tag":"haiku","prompt":"Say hello"}' | jq .

# Test as Anthropic client
curl -s http://127.0.0.1:${config.port}/anthropic/v1/messages \\
  -H 'Content-Type: application/json' \\
  -H 'x-api-key: test' \\
  -d '{"model":"sonnet","max_tokens":64,"messages":[{"role":"user","content":"Hi"}]}' | jq .`}
        </pre>
      </Card>
    </div>
  );
}

function RouteForm({ initial, providers, tags, onSave, onCancel }: {
  initial?: Route;
  providers: string[];
  tags: string[];
  onSave: (route: Route) => void;
  onCancel: () => void;
}) {
  const [endpoint, setEndpoint] = useState(initial?.endpoint || '/v1/chat/completions');
  const [model, setModel] = useState(initial?.model || '');
  const [provider, setProvider] = useState(initial?.provider || providers[0] || '');
  const [selectedTags, setSelectedTags] = useState<string[]>(initial?.tags || []);
  const [format, setFormat] = useState<'openai' | 'anthropic'>(initial?.format || 'openai');

  const toggleTag = (tag: string) => {
    setSelectedTags(prev => prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]);
  };

  const valid = endpoint && model && provider && selectedTags.length > 0;

  return (
    <Card style={{ marginBottom: 16, background: 'var(--bg-input)' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <Input label="Endpoint" value={endpoint} onChange={e => setEndpoint(e.target.value)} />
        <Input label="Model" value={model} onChange={e => setModel(e.target.value)} />
        <Select label="Provider" value={provider} onChange={e => setProvider(e.target.value)}>
          {providers.map(p => <option key={p} value={p}>{p}</option>)}
        </Select>
        <Select label="Format" value={format} onChange={e => setFormat(e.target.value as 'openai' | 'anthropic')}>
          <option value="openai">OpenAI</option>
          <option value="anthropic">Anthropic</option>
        </Select>
        <div style={{ gridColumn: '1 / -1' }}>
          <label style={{ display: 'block', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 6, fontWeight: 500 }}>Tags</label>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {tags.map(tag => (
              <button key={tag} onClick={() => toggleTag(tag)} style={{
                padding: '3px 12px', borderRadius: 'var(--radius-full)', fontSize: 12, cursor: 'pointer',
                background: selectedTags.includes(tag) ? 'var(--accent)' : '#2a2a2a',
                color: selectedTags.includes(tag) ? '#fff' : 'var(--text-secondary)',
                border: 'none', fontWeight: 500,
              }}>
                {tag}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div style={{ marginTop: 14, display: 'flex', gap: 8 }}>
        <Button variant="primary" disabled={!valid} onClick={() => onSave({ endpoint, model, provider, tags: selectedTags, format })}>Save</Button>
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
      </div>
    </Card>
  );
}
