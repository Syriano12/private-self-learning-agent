import * as SecureStore from 'expo-secure-store';
import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

const TOKEN_KEY = 'private-agent.owner-token';
const BASE_URL_KEY = 'private-agent.api-base-url';

export type SessionConfig = {
  apiBaseUrl: string;
  token: string;
};

type SessionContextValue = SessionConfig & {
  ready: boolean;
  save: (config: SessionConfig) => Promise<void>;
  clear: () => Promise<void>;
};

const SessionContext = createContext<SessionContextValue | null>(null);

export function AgentSessionProvider({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false);
  const [apiBaseUrl, setApiBaseUrl] = useState(process.env.EXPO_PUBLIC_AGENT_API_BASE_URL ?? '');
  const [token, setToken] = useState('');

  useEffect(() => {
    let cancelled = false;
    Promise.all([SecureStore.getItemAsync(BASE_URL_KEY), SecureStore.getItemAsync(TOKEN_KEY)])
      .then(([storedBase, storedToken]) => {
        if (cancelled) return;
        if (storedBase) setApiBaseUrl(storedBase);
        if (storedToken) setToken(storedToken);
      })
      .finally(() => {
        if (!cancelled) setReady(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const save = useCallback(async (config: SessionConfig) => {
    const nextBase = config.apiBaseUrl.trim();
    const nextToken = config.token.trim();
    await SecureStore.setItemAsync(BASE_URL_KEY, nextBase);
    await SecureStore.setItemAsync(TOKEN_KEY, nextToken, { keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY });
    setApiBaseUrl(nextBase);
    setToken(nextToken);
  }, []);

  const clear = useCallback(async () => {
    await SecureStore.deleteItemAsync(TOKEN_KEY);
    setToken('');
  }, []);

  const value = useMemo(() => ({ ready, apiBaseUrl, token, save, clear }), [ready, apiBaseUrl, token, save, clear]);
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useAgentSession(): SessionContextValue {
  const value = useContext(SessionContext);
  if (!value) throw new Error('AgentSessionProvider is missing');
  return value;
}
