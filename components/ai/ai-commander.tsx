'use client';

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, Send, Minimize2, Sparkles, Command } from 'lucide-react';
import { useUIStore } from '@/lib/store/ui-store';
import { interpretCommand } from '@/lib/ai/commander-service';
import { useRouter } from 'next/navigation';

export default function AICommander() {
  const [isOpen, setIsOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [messages, setMessages] = useState<{ role: 'user' | 'ai'; text: string }[]>([
    { role: 'ai', text: 'Commands ready. Use natural language to control the environment.' }
  ]);

  const { currentScene, setScene, setTheme } = useUIStore();
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  const handleCommand = async () => {
    if (!inputValue.trim()) return;

    // Add user message
    const userText = inputValue;
    setMessages(prev => [...prev, { role: 'user', text: userText }]);
    setInputValue('');
    setIsProcessing(true);

    // Simulate AI thinking "feel"
    setTimeout(async () => {
      const command = await interpretCommand(userText);

      switch (command.type) {
        case 'CHANGE_SCENE':
          setScene(command.payload);
          setMessages(prev => [...prev, { role: 'ai', text: `Switching scene to: ${command.payload.replace('_', ' ')}` }]);
          break;
        case 'CHANGE_THEME':
          setTheme(command.payload);
          setMessages(prev => [...prev, { role: 'ai', text: `Applying ${command.payload} theme.` }]);
          break;
        case 'NAVIGATE':
          router.push(command.payload);
          setMessages(prev => [...prev, { role: 'ai', text: `Navigating to ${command.payload}...` }]);
          break;
        case 'UNKNOWN':
          setMessages(prev => [...prev, { role: 'ai', text: command.payload }]);
          break;
      }
      setIsProcessing(false);
    }, 800);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleCommand();
  };

  return (
    <>
      <div className="fixed bottom-8 right-8 z-50 flex flex-col items-end gap-4 pointer-events-none">

        {/* Chat Interface */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.95 }}
              className="pointer-events-auto bg-black/80 backdrop-blur-xl border border-white/10 rounded-2xl w-[350px] overflow-hidden shadow-2xl"
            >
              {/* Header */}
              <div className="flex items-center justify-between p-4 border-b border-white/5 bg-white/5">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-[#D4AF37]" />
                  <span className="text-sm font-semibold text-white">System AI</span>
                </div>
                <button onClick={() => setIsOpen(false)} className="text-white/40 hover:text-white">
                  <Minimize2 className="w-4 h-4" />
                </button>
              </div>

              {/* Messages */}
              <div className="h-[250px] overflow-y-auto p-4 space-y-4">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div
                      className={`max-w-[80%] p-3 rounded-lg text-sm ${msg.role === 'user'
                          ? 'bg-[#D4AF37]/20 text-white border border-[#D4AF37]/30'
                          : 'bg-white/10 text-white/80'
                        }`}
                    >
                      {msg.text}
                    </div>
                  </div>
                ))}
                {isProcessing && (
                  <div className="flex justify-start">
                    <div className="bg-white/10 p-3 rounded-lg flex gap-1">
                      <span className="w-1.5 h-1.5 bg-white/50 rounded-full animate-bounce" />
                      <span className="w-1.5 h-1.5 bg-white/50 rounded-full animate-bounce delay-75" />
                      <span className="w-1.5 h-1.5 bg-white/50 rounded-full animate-bounce delay-150" />
                    </div>
                  </div>
                )}
              </div>

              {/* Input */}
              <div className="p-3 border-t border-white/10 bg-black/40">
                <div className="relative flex items-center">
                  <input
                    ref={inputRef}
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask AI to change background..."
                    className="w-full bg-white/5 border border-white/10 rounded-xl pl-4 pr-10 py-2.5 text-sm text-white focus:outline-none focus:border-[#D4AF37]/50 transition-colors placeholder:text-white/20"
                    autoFocus
                  />
                  <button
                    onClick={handleCommand}
                    className="absolute right-2 p-1.5 bg-[#D4AF37] rounded-lg text-black hover:bg-[#F4CF57] transition-colors"
                  >
                    <Send className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Toggle Trigger */}
        <motion.button
          onClick={() => setIsOpen(!isOpen)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="pointer-events-auto group relative flex items-center justify-center w-14 h-14 rounded-full bg-gradient-to-br from-[#D4AF37] to-[#8C7335] shadow-lg shadow-[#D4AF37]/20 border border-white/20"
        >
          <div className="absolute inset-0 rounded-full bg-white/20 animate-pulse group-hover:animate-ping opacity-20" />
          {isOpen ? (
            <Minimize2 className="w-6 h-6 text-black" />
          ) : (
            <Command className="w-6 h-6 text-black" />
          )}
        </motion.button>
      </div>
    </>
  );
}
