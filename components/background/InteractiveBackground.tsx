'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useUIStore } from '@/lib/store/ui-store';

/* ═══════════════════════════════════════════════════════════════════════════
   INTERACTIVE BACKGROUND - CSS + Canvas Version
   Mouse-reactive particle field with flowing motion - No Three.js dependency
   Inspired by igloo.inc, buttermax.net, drumspirit.be
   ═══════════════════════════════════════════════════════════════════════════ */

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  color: string;
  opacity: number;
  originalX: number;
  originalY: number;
}

export default function InteractiveBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: 0, y: 0 });
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number>(0);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const { currentScene } = useUIStore();

  const getSceneConfig = () => {
    switch (currentScene) {
      case 'romantic_dinner':
        return {
          filter: 'hue-rotate(320deg) contrast(1.2) saturate(1.5)',
          overlay: 'linear-gradient(to bottom, rgba(50,0,20,0.5), rgba(10,0,10,0.8))',
          orbColor1: '#ff0055',
          orbColor2: '#ff99cc'
        };
      case 'war_room':
        return {
          filter: 'sepia(1) hue-rotate(-50deg) contrast(2) saturate(2) brightness(0.8)',
          overlay: 'radial-gradient(circle, transparent, rgba(80,0,0,0.9))',
          orbColor1: '#ff0000',
          orbColor2: '#800000'
        };
      case 'cyber_void':
        return {
          filter: 'hue-rotate(90deg) contrast(2) invert(0.9)',
          overlay: 'linear-gradient(0deg, rgba(0,20,0,0.9) 0%, transparent 100%)',
          orbColor1: '#00ff00',
          orbColor2: '#003300'
        };
      case 'harvey_office':
      default:
        return {
          filter: 'saturate(1.3) contrast(1.15) brightness(1.1)',
          overlay: 'radial-gradient(ellipse at center, transparent 35%, rgba(10,10,15,0.65) 100%)',
          orbColor1: '#D4AF37',
          orbColor2: '#3B82F6'
        };
    }
  };
  const config = getSceneConfig();

  // Initialize particles
  const initParticles = useCallback((width: number, height: number) => {
    const particles: Particle[] = [];
    const particleCount = Math.floor((width * height) / 8000); // Responsive particle count

    const colors = [
      'rgba(212, 175, 55, 0.6)',  // Gold
      'rgba(212, 175, 55, 0.4)',  // Gold light
      'rgba(59, 130, 246, 0.5)',  // Blue
      'rgba(59, 130, 246, 0.3)',  // Blue light
      'rgba(139, 92, 246, 0.3)',  // Purple
      'rgba(255, 255, 255, 0.2)', // White
    ];

    for (let i = 0; i < particleCount; i++) {
      const x = Math.random() * width;
      const y = Math.random() * height;
      particles.push({
        x,
        y,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        size: Math.random() * 2 + 0.5,
        color: colors[Math.floor(Math.random() * colors.length)],
        opacity: Math.random() * 0.5 + 0.2,
        originalX: x,
        originalY: y,
      });
    }
    particlesRef.current = particles;
  }, []);

  // Animation loop
  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = canvas;
    const mouse = mouseRef.current;

    // Clear with fade effect
    ctx.fillStyle = 'rgba(10, 10, 15, 0.15)';
    ctx.fillRect(0, 0, width, height);

    const particles = particlesRef.current;
    const time = Date.now() * 0.001;

    // Draw connections first (behind particles)
    ctx.beginPath();
    for (let i = 0; i < particles.length; i++) {
      const p1 = particles[i];
      for (let j = i + 1; j < particles.length; j++) {
        const p2 = particles[j];
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 120) {
          ctx.strokeStyle = `rgba(212, 175, 55, ${0.1 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
        }
      }
    }
    ctx.stroke();

    // Update and draw particles
    particles.forEach((particle) => {
      // Wave motion
      const waveX = Math.sin(time + particle.originalX * 0.01) * 2;
      const waveY = Math.cos(time + particle.originalY * 0.01) * 2;

      // Mouse influence
      const dx = mouse.x - particle.x;
      const dy = mouse.y - particle.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const maxDist = 200;

      if (dist < maxDist) {
        const force = (maxDist - dist) / maxDist;
        const angle = Math.atan2(dy, dx);
        // Particles are gently attracted to mouse
        particle.vx += Math.cos(angle) * force * 0.02;
        particle.vy += Math.sin(angle) * force * 0.02;
      }

      // Return to original position slowly
      particle.vx += (particle.originalX - particle.x + waveX) * 0.001;
      particle.vy += (particle.originalY - particle.y + waveY) * 0.001;

      // Apply velocity with friction
      particle.x += particle.vx;
      particle.y += particle.vy;
      particle.vx *= 0.98;
      particle.vy *= 0.98;

      // Draw particle with glow
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
      ctx.fillStyle = particle.color;
      ctx.fill();

      // Glow effect for larger particles
      if (particle.size > 1.5) {
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size * 3, 0, Math.PI * 2);
        const gradient = ctx.createRadialGradient(
          particle.x, particle.y, 0,
          particle.x, particle.y, particle.size * 3
        );
        gradient.addColorStop(0, particle.color.replace('0.', '0.2'));
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.fill();
      }
    });

    // Draw floating geometric shapes
    drawGeometricShapes(ctx, width, height, time, mouse);

    animationRef.current = requestAnimationFrame(animate);
  }, []);

  // Draw wireframe geometric shapes that react to mouse
  const drawGeometricShapes = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    time: number,
    mouse: { x: number; y: number }
  ) => {
    // Shape 1: Rotating icosahedron-like shape
    const shape1X = width * 0.75 + Math.sin(time * 0.5) * 50 + (mouse.x - width / 2) * 0.02;
    const shape1Y = height * 0.3 + Math.cos(time * 0.3) * 30 + (mouse.y - height / 2) * 0.02;
    drawWireframeSphere(ctx, shape1X, shape1Y, 80, time * 0.5, 'rgba(212, 175, 55, 0.3)');

    // Shape 2: Smaller rotating shape
    const shape2X = width * 0.2 + Math.cos(time * 0.4) * 40 - (mouse.x - width / 2) * 0.015;
    const shape2Y = height * 0.7 + Math.sin(time * 0.6) * 25 - (mouse.y - height / 2) * 0.015;
    drawWireframeSphere(ctx, shape2X, shape2Y, 50, -time * 0.3, 'rgba(59, 130, 246, 0.25)');

    // Shape 3: Floating ring
    const shape3X = width * 0.5 + Math.sin(time * 0.2) * 100;
    const shape3Y = height * 0.15 + Math.cos(time * 0.4) * 20;
    drawWireframeRing(ctx, shape3X, shape3Y, 60, time * 0.2, 'rgba(139, 92, 246, 0.2)');
  };

  const drawWireframeSphere = (
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radius: number,
    rotation: number,
    color: string
  ) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;

    // Draw latitude lines
    for (let i = 0; i < 6; i++) {
      const angle = (i / 6) * Math.PI;
      const r = radius * Math.sin(angle);
      const y = cy + radius * Math.cos(angle) * 0.3;

      ctx.beginPath();
      ctx.ellipse(cx, y, r, r * 0.3, rotation, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw longitude lines
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2 + rotation;
      ctx.beginPath();
      ctx.ellipse(cx, cy, radius * 0.1, radius, angle, 0, Math.PI * 2);
      ctx.stroke();
    }
  };

  const drawWireframeRing = (
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radius: number,
    rotation: number,
    color: string
  ) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.ellipse(cx, cy, radius, radius * 0.25, rotation, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(cx, cy, radius * 0.8, radius * 0.2, rotation + 0.5, 0, Math.PI * 2);
    ctx.stroke();
  };

  // Handle mouse move
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      setDimensions({ width, height });

      if (canvasRef.current) {
        canvasRef.current.width = width;
        canvasRef.current.height = height;
      }

      initParticles(width, height);
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [initParticles]);

  // Start animation
  useEffect(() => {
    if (dimensions.width > 0) {
      animate();
    }
    return () => cancelAnimationFrame(animationRef.current);
  }, [animate, dimensions]);

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        width: '100vw',
        height: '100vh',
        zIndex: -10,
        overflow: 'hidden',
        backgroundColor: '#0A0A0F',
        transition: 'all 1s ease-in-out',
      }}
    >
      {/* Dynamic Background Image Layer */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          backgroundImage: 'url(/harvey-bg.png)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
          opacity: 0.75,
          filter: config.filter,
          transform: 'scale(1.1)',
          transition: 'filter 1.5s ease',
        }}
      />

      {/* Dynamic Overlay */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          background: config.overlay,
          pointerEvents: 'none',
          transition: 'background 1.5s ease',
        }}
      />

      {/* Bottom fade */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          background: 'linear-gradient(to top, rgba(10,10,15,0.8) 0%, transparent 60%)',
          pointerEvents: 'none',
        }}
      />

      {/* Canvas Layer */}
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          mixBlendMode: 'screen',
          background: 'transparent',
        }}
      />

      {/* Dynamic Orbs */}
      <motion.div
        animate={{
          x: [0, 50, 0],
          y: [0, 30, 0],
          scale: [1, 1.1, 1],
          background: `radial-gradient(circle, ${config.orbColor1} 0%, transparent 70%)`
        }}
        transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          position: 'absolute',
          width: '600px',
          height: '600px',
          borderRadius: '50%',
          filter: 'blur(150px)',
          opacity: 0.3,
          pointerEvents: 'none',
          left: '20%',
          top: '30%',
        }}
      />
      <motion.div
        animate={{
          x: [0, -40, 0],
          y: [0, -20, 0],
          scale: [1, 1.15, 1],
          background: `radial-gradient(circle, ${config.orbColor2} 0%, transparent 70%)`
        }}
        transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          position: 'absolute',
          width: '500px',
          height: '500px',
          borderRadius: '50%',
          filter: 'blur(120px)',
          opacity: 0.2,
          pointerEvents: 'none',
          right: '15%',
          bottom: '25%',
        }}
      />
    </div>
  );
}
