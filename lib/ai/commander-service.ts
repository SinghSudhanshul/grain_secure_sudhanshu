import { BackgroundScene, UITheme } from '@/lib/store/ui-store';

export type SystemCommand =
  | { type: 'CHANGE_SCENE'; payload: BackgroundScene }
  | { type: 'CHANGE_THEME'; payload: UITheme }
  | { type: 'NAVIGATE'; payload: string }
  | { type: 'UNKNOWN'; payload: string };

export async function interpretCommand(text: string): Promise<SystemCommand> {
  const lower = text.toLowerCase();

  // "Harvey kiss Donna" -> Romantic Scene
  if (lower.includes('kiss') || lower.includes('date') || lower.includes('love') || lower.includes('romance')) {
    return { type: 'CHANGE_SCENE', payload: 'romantic_dinner' };
  }

  // "Harvey beat Luis" -> War Room / Conflict
  if (lower.includes('beat') || lower.includes('fight') || lower.includes('punch') || lower.includes('war')) {
    return { type: 'CHANGE_SCENE', payload: 'war_room' };
  }

  // "Work mode" / "Office"
  if (lower.includes('work') || lower.includes('office') || lower.includes('focus')) {
    return { type: 'CHANGE_SCENE', payload: 'harvey_office' };
  }

  // "Hacker mode" / "Data"
  if (lower.includes('hack') || lower.includes('data') || lower.includes('code') || lower.includes('matrix')) {
    return { type: 'CHANGE_SCENE', payload: 'cyber_void' };
  }

  // Navigation
  if (lower.includes('dashboard') || lower.includes('stats')) return { type: 'NAVIGATE', payload: '/dashboard' };
  if (lower.includes('demo') || lower.includes('model')) return { type: 'NAVIGATE', payload: '/demo' };
  if (lower.includes('login') || lower.includes('sign in')) return { type: 'NAVIGATE', payload: '/login' };

  // Themes
  if (lower.includes('gold') || lower.includes('luxury')) return { type: 'CHANGE_THEME', payload: 'gold' };
  if (lower.includes('blue') || lower.includes('tech')) return { type: 'CHANGE_THEME', payload: 'cyber_blue' };
  if (lower.includes('red') || lower.includes('alert')) return { type: 'CHANGE_THEME', payload: 'alert_red' };

  return { type: 'UNKNOWN', payload: "I'm not sure how to do that yet, sir." };
}
