import type { ExpoConfig } from 'expo/config';

const config: ExpoConfig = {
  name: 'Private Agent Control',
  slug: 'private-agent-control',
  version: '0.1.0',
  orientation: 'portrait',
  scheme: 'privateagent',
  userInterfaceStyle: 'dark',
  newArchEnabled: true,
  experiments: {
    typedRoutes: true,
  },
  plugins: ['expo-router', 'expo-secure-store'],
  android: {
    package: 'com.privateagent.control',
    adaptiveIcon: {
      backgroundColor: '#071018',
    },
  },
  extra: {
    apiBaseUrl: process.env.EXPO_PUBLIC_AGENT_API_BASE_URL ?? '',
  },
};

export default config;
