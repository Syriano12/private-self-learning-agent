import { Stack } from 'expo-router';
import React from 'react';

import { AgentSessionProvider } from '@/lib/session';

export default function RootLayout() {
  return (
    <AgentSessionProvider>
      <Stack screenOptions={{ headerShown: false }} />
    </AgentSessionProvider>
  );
}
