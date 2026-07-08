import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Card, Action } from '@/types/game';
import { UnoCard } from './UnoCard';

interface PlayerHandProps {
  cards: Card[];
  legalActions: Action[];
  onPlayCard: (actionIdx: number) => void;
  disabled?: boolean;
}

export const PlayerHand: React.FC<PlayerHandProps> = ({
  cards,
  legalActions,
  onPlayCard,
  disabled = false,
}) => {
  // Build a set of legal action indices for quick lookup
  const legalSet = new Set(legalActions.map((a) => a.action_idx));

  // Map cards to their playable actions
  // A card can match multiple actions (e.g., wild cards have 4 color variants)
  const getCardActions = (card: Card): Action[] => {
    return legalActions.filter((action) => {
      // Exact card match for colored cards
      if (action.action_idx === card.card_idx) return true;
      // Wild cards: actions 52-55 correspond to physical wild (idx 52)
      if (card.card_idx === 52 && action.action_idx >= 52 && action.action_idx <= 55) return true;
      // WD4: actions 56-59 correspond to physical WD4 (idx 53)
      if (card.card_idx === 53 && action.action_idx >= 56 && action.action_idx <= 59) return true;
      return false;
    });
  };

  // Calculate fan layout
  const totalCards = cards.length;
  const maxAngle = Math.min(totalCards * 3, 30);

  return (
    <div className="flex justify-center items-end py-4 min-h-[160px]">
      <AnimatePresence mode="popLayout">
        {cards.map((card, index) => {
          const cardActions = getCardActions(card);
          const isPlayable = !disabled && cardActions.length > 0;
          const angle = totalCards > 1
            ? -maxAngle / 2 + (maxAngle / (totalCards - 1)) * index
            : 0;
          const yOffset = Math.abs(angle) * 0.5;

          return (
            <motion.div
              key={`${card.card_idx}-${index}`}
              layout
              initial={{ y: 100, opacity: 0 }}
              animate={{
                y: yOffset,
                opacity: 1,
                rotate: angle,
              }}
              exit={{ y: -100, opacity: 0, scale: 0.5 }}
              transition={{ type: 'spring', stiffness: 300, damping: 25 }}
              style={{
                marginLeft: index > 0 ? '-0.75rem' : 0,
                zIndex: index,
              }}
            >
              <UnoCard
                card={card}
                isPlayable={isPlayable}
                onClick={() => {
                  if (cardActions.length === 1) {
                    onPlayCard(cardActions[0].action_idx);
                  }
                  // For wilds with multiple colors, let the parent handle the picker
                  if (cardActions.length > 1) {
                    // Use the first action as default — the ColorPicker will override
                    onPlayCard(cardActions[0].action_idx);
                  }
                }}
                animationDelay={index * 0.05}
              />
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
};
