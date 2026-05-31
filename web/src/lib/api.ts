const API_BASE = '/api';

export interface Provider {
  name: string;
  base_url: string;
  api_key: string;
  auth_type: 'bearer' | 'x_api_key' | 'x_goog_api_key';
}

export interface Route {
  endpoint: string;
  model: string;
  provider: string;
  tags: string[];
  format: 'openai' | 'anthropic';
}

export interface Tag {
  name: string;
  color: string;
  is_auto: boolean;
}

export interface AppConfig {
  port: number;
  providers: Record<string, Provider>;
  routes: Route[];
  tags: Tag[];
  current_tag: string;
}

export interface Status {
  current_tag: string;
  takeover: {
    active: boolean;
    proxy_url: string | null;
  };
}

export interface RequestLog {
  request_model: string;
  tag: string;
  provider: string;
  target_model: string;
  timestamp: string;
}

export async function getConfig(): Promise<AppConfig> {
  const res = await fetch(`${API_BASE}/config`);
  return res.json();
}

export async function updateConfig(config: AppConfig): Promise<void> {
  await fetch(`${API_BASE}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
}

export async function setCurrentTag(tag: string): Promise<void> {
  await fetch(`${API_BASE}/current-tag`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tag }),
  });
}

export async function getStatus(): Promise<Status> {
  const res = await fetch(`${API_BASE}/status`);
  return res.json();
}

export async function takeoverClaude(): Promise<{ proxy_url: string }> {
  const res = await fetch(`${API_BASE}/takeover/claude`, { method: 'POST' });
  return res.json();
}

export async function restoreClaude(): Promise<void> {
  await fetch(`${API_BASE}/takeover/claude`, { method: 'DELETE' });
}

export async function getLogs(): Promise<RequestLog[]> {
  const res = await fetch(`${API_BASE}/logs`);
  return res.json();
}

export interface TestResult {
  success: boolean;
  tag: string;
  provider: string;
  model: string;
  format: string;
  latency_ms: number;
  error: string | null;
  response: any | null;
}

export async function testRoute(tag: string, prompt?: string): Promise<TestResult> {
  const res = await fetch(`${API_BASE}/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tag, prompt: prompt || 'Hi, reply with one word.' }),
  });
  return res.json();
}
