import { create } from 'zustand';

export type BackgroundScene = 'harvey_office' | 'data_center' | 'war_room' | 'cyber_void' | 'romantic_dinner'; // "romantic_dinner" for the "kiss" request
export type UITheme = 'gold' | 'cyber_blue' | 'alert_red' | 'midnight_purple';

interface UIState {
  currentScene: BackgroundScene;
  theme: UITheme;
  isAIActive: boolean;
  aiCommandHistory: string[];

  // Actions
  setScene: (scene: BackgroundScene) => void;
  setTheme: (theme: UITheme) => void;
  toggleAI: () => void;
  addCommand: (command: string) => void;
}

export const useUIStore = create<UIState>((set) => ({
  currentScene: 'harvey_office',
  theme: 'gold',
  isAIActive: false,
  aiCommandHistory: [],

  setScene: (scene) => set({ currentScene: scene }),
  setTheme: (theme) => set({ theme }),
  toggleAI: () => set((state) => ({ isAIActive: !state.isAIActive })),
  addCommand: (cmd) => set((state) => ({ aiCommandHistory: [cmd, ...state.aiCommandHistory].slice(0, 10) })),
}));
