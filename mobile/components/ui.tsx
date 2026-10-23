import React from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';

export const colors = {
  bg: '#071018',
  panel: '#0D1B24',
  panelAlt: '#112631',
  border: '#20404A',
  text: '#EAF4F5',
  muted: '#8EA9AE',
  accent: '#3CE0C2',
  warning: '#F2C66D',
  danger: '#FF8E88',
  blue: '#87B7FF',
};

export function Card({ children, style }: { children: React.ReactNode; style?: object }) {
  return <View style={[styles.card, style]}>{children}</View>;
}

export function Label({ children }: { children: React.ReactNode }) {
  return <Text style={styles.label}>{children}</Text>;
}

export function Button({ label, onPress, disabled = false, variant = 'primary', accessibilityLabel }: { label: string; onPress: () => void; disabled?: boolean; variant?: 'primary' | 'quiet' | 'danger'; accessibilityLabel?: string }) {
  return (
    <Pressable accessibilityRole="button" accessibilityLabel={accessibilityLabel ?? label} disabled={disabled} onPress={onPress} style={({ pressed }) => [styles.button, variant === 'quiet' && styles.quietButton, variant === 'danger' && styles.dangerButton, pressed && styles.pressed, disabled && styles.disabled]}>
      <Text style={[styles.buttonText, variant === 'quiet' && styles.quietButtonText]}>{label}</Text>
    </Pressable>
  );
}

export function Field({ label, value, onChangeText, placeholder, secureTextEntry = false, autoCapitalize = 'none' }: { label: string; value: string; onChangeText: (value: string) => void; placeholder?: string; secureTextEntry?: boolean; autoCapitalize?: 'none' | 'sentences' | 'words' | 'characters' }) {
  return (
    <View style={styles.field}>
      <Label>{label}</Label>
      <TextInput accessibilityLabel={label} autoCapitalize={autoCapitalize} autoCorrect={false} placeholder={placeholder} placeholderTextColor={colors.muted} secureTextEntry={secureTextEntry} style={styles.input} value={value} onChangeText={onChangeText} />
    </View>
  );
}

export function StateMessage({ title, detail, loading = false }: { title: string; detail?: string; loading?: boolean }) {
  return <View style={styles.state}>{loading ? <ActivityIndicator color={colors.accent} /> : null}<Text style={styles.stateTitle}>{title}</Text>{detail ? <Text style={styles.stateDetail}>{detail}</Text> : null}</View>;
}

export function statusColor(status: string): string {
  if (['COMPLETED', 'VERIFIED', 'APPROVED', 'running', 'ready'].includes(status)) return colors.accent;
  if (['FAILED', 'BLOCKED', 'EXECUTION_UNKNOWN', 'DENIED', 'fatal'].includes(status)) return colors.danger;
  if (['WAITING_APPROVAL', 'PENDING', 'stopped', 'not_ready'].includes(status)) return colors.warning;
  return colors.blue;
}

const styles = StyleSheet.create({
  card: { backgroundColor: colors.panel, borderColor: colors.border, borderRadius: 18, borderWidth: 1, gap: 9, padding: 16 },
  label: { color: colors.muted, fontSize: 12, fontWeight: '700', letterSpacing: 0.7, textTransform: 'uppercase' },
  button: { alignItems: 'center', backgroundColor: colors.accent, borderRadius: 12, minHeight: 44, justifyContent: 'center', paddingHorizontal: 16 },
  quietButton: { backgroundColor: colors.panelAlt, borderColor: colors.border, borderWidth: 1 },
  dangerButton: { backgroundColor: colors.danger },
  pressed: { opacity: 0.72 },
  disabled: { opacity: 0.4 },
  buttonText: { color: colors.bg, fontSize: 14, fontWeight: '800' },
  quietButtonText: { color: colors.text },
  field: { gap: 7 },
  input: { backgroundColor: colors.panelAlt, borderColor: colors.border, borderRadius: 12, borderWidth: 1, color: colors.text, minHeight: 46, paddingHorizontal: 12 },
  state: { alignItems: 'center', gap: 8, paddingHorizontal: 20, paddingVertical: 40 },
  stateTitle: { color: colors.text, fontSize: 16, fontWeight: '800', textAlign: 'center' },
  stateDetail: { color: colors.muted, fontSize: 13, lineHeight: 20, textAlign: 'center' },
});
