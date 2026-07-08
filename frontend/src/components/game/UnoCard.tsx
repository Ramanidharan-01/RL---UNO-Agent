import React from 'react';
import { motion } from 'framer-motion';
import type { Card } from '@/types/game';

interface UnoCardProps {
  card: Card;
  isPlayable?: boolean;
  onClick?: () => void;
  size?: 'sm' | 'md' | 'lg';
  faceDown?: boolean;
  animationDelay?: number;
}

const colorMap: Record<string, string> = {
  red: 'uno-card-red',
  yellow: 'uno-card-yellow',
  green: 'uno-card-green',
  blue: 'uno-card-blue',
  wild: 'uno-card-wild',
};

function getCardColor(name: string): string {
  if (name.startsWith('wild')) return 'uno-card-wild';
  const color = name.split(':')[0];
  return colorMap[color] || 'uno-card-wild';
}

function getCardLabel(name: string): { top: string; center: string } {
  if (name === 'wild') return { top: 'W', center: '🌈' };
  if (name === 'wild_draw4') return { top: 'W', center: '+4' };
  if (name.startsWith('wild->')) {
    const targetColor = name.split('->')[1];
    return { top: 'W', center: targetColor.charAt(0).toUpperCase() };
  }
  if (name.startsWith('wild_draw4->')) {
    const targetColor = name.split('->')[1];
    return { top: '+4', center: targetColor.charAt(0).toUpperCase() };
  }

  const parts = name.split(':');
  if (parts.length === 2) {
    const rank = parts[1];
    const display: Record<string, string> = {
      skip: '⊘', reverse: '⇄', draw2: '+2',
    };
    return { top: display[rank] || rank, center: display[rank] || rank };
  }

  return { top: '?', center: '?' };
}

const sizeClasses = {
  sm: 'w-14 h-20 text-xs',
  md: 'w-20 h-28 text-sm',
  lg: 'w-28 h-40 text-base',
};

export const UnoCard: React.FC<UnoCardProps> = ({
  card,
  isPlayable = false,
  onClick,
  size = 'md',
  faceDown = false,
  animationDelay = 0,
}) => {
  const colorClass = getCardColor(card.name);
  const label = getCardLabel(card.name);

  if (faceDown) {
    return (
      <motion.div
        initial={{ rotateY: 180, opacity: 0 }}
        animate={{ rotateY: 0, opacity: 1 }}
        transition={{ delay: animationDelay, duration: 0.3 }}
        className={`uno-card ${sizeClasses[size]} bg-gradient-to-br from-surface-700 to-surface-800 border-2 border-surface-600 flex items-center justify-center`}
      >
        <div className="text-2xl font-display font-bold text-accent opacity-60">U</div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ y: -50, opacity: 0, rotate: -10 }}
      animate={{ y: 0, opacity: 1, rotate: 0 }}
      transition={{ delay: animationDelay, duration: 0.3, type: 'spring', stiffness: 200 }}
      whileHover={isPlayable ? { y: -12, scale: 1.08 } : undefined}
      whileTap={isPlayable ? { scale: 0.95 } : undefined}
      onClick={isPlayable ? onClick : undefined}
      className={`
        uno-card ${sizeClasses[size]} ${colorClass}
        ${isPlayable ? 'uno-card-playable cursor-pointer' : 'cursor-default opacity-85'}
        flex flex-col items-center justify-between p-1.5 text-white font-bold
      `}
    >
      <span className="self-start text-[0.65em] leading-none">{label.top}</span>
      <span className="text-[1.8em] leading-none font-display">{label.center}</span>
      <span className="self-end text-[0.65em] leading-none rotate-180">{label.top}</span>
    </motion.div>
  );
};
