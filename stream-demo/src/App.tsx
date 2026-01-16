import React, { useState, useCallback, useRef, useEffect } from 'react';
import { StreamingTTS } from './streaming-tts';

const LOREM_IPSUM = `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit, nec luctus magna felis sollicitudin mauris. Integer in mauris eu nibh euismod gravida.

Duis ac tellus et risus vulputate vehicula. Donec lobortis risus a elit. Etiam tempor ultrices risus. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.`;

/**
 * Find word boundary for clean highlighting (copied from voice-assist).
 */
function findWordBoundary(text: string, charIndex: number): number {
  if (charIndex <= 0) return 0;
  if (charIndex >= text.length) return text.length;
  if (text[charIndex] === ' ') return charIndex + 1;
  if (charIndex > 0 && text[charIndex - 1] === ' ') return charIndex;
  const prevSpace = text.lastIndexOf(' ', charIndex - 1);
  return prevSpace > 0 ? prevSpace + 1 : 0;
}

/**
 * Find position N words ahead of given position.
 */
function findWordsAhead(text: string, charIndex: number, numWords: number): number {
  let pos = charIndex;
  let wordsFound = 0;
  while (pos < text.length && wordsFound < numWords) {
    // Skip to next space
    while (pos < text.length && text[pos] !== ' ') pos++;
    // Skip spaces
    while (pos < text.length && text[pos] === ' ') pos++;
    wordsFound++;
  }
  return pos;
}

export function App() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [playbackPosition, setPlaybackPosition] = useState(0);
  const [hasBeenPlayed, setHasBeenPlayed] = useState(false);
  const [voice, setVoice] = useState('cosette');

  const ttsRef = useRef<StreamingTTS | null>(null);
  const lastBoundaryRef = useRef(-1);
  const textRef = useRef(LOREM_IPSUM);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      ttsRef.current?.stop();
    };
  }, []);

  const handlePlay = useCallback(() => {
    if (isSpeaking) return;

    const tts = new StreamingTTS();
    ttsRef.current = tts;
    lastBoundaryRef.current = -1;

    setIsSpeaking(true);
    setPlaybackPosition(0);

    tts.start(textRef.current, { voice }, {
      onStart: () => {
        console.log('[TTS] Started');
      },
      onProgress: (elapsed, total) => {
        // Use elapsed time directly with estimated speaking rate
        // Average speaking rate is ~150 wpm = ~12.5 chars/sec (assuming 5 chars/word)
        const CHARS_PER_SECOND = 12;
        const estimatedChar = Math.floor(elapsed * CHARS_PER_SECOND);

        // Snap to word boundary
        const wordBoundary = findWordBoundary(textRef.current, estimatedChar);

        // Clamp to text length
        const clampedBoundary = Math.min(wordBoundary, textRef.current.length);

        // Only update if boundary changed
        if (clampedBoundary !== lastBoundaryRef.current) {
          lastBoundaryRef.current = clampedBoundary;
          setPlaybackPosition(clampedBoundary);
        }
      },
      onEnd: (completed) => {
        console.log('[TTS] Ended, completed:', completed);
        setIsSpeaking(false);
        if (completed) {
          setPlaybackPosition(textRef.current.length);
          setHasBeenPlayed(true);
        }
      },
      onError: (error) => {
        console.error('[TTS] Error:', error);
        setIsSpeaking(false);
      },
    }).catch((err) => {
      console.error('[TTS] Start error:', err);
      setIsSpeaking(false);
    });
  }, [isSpeaking, voice]);

  const handleStop = useCallback(() => {
    ttsRef.current?.stop();
    setIsSpeaking(false);
  }, []);

  const handleClick = useCallback(() => {
    if (isSpeaking) {
      handleStop();
    } else {
      handlePlay();
    }
  }, [isSpeaking, handlePlay, handleStop]);

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    // Reset and replay
    setPlaybackPosition(0);
    setHasBeenPlayed(false);
    handleStop();
    // Small delay then play
    setTimeout(handlePlay, 100);
  }, [handlePlay, handleStop]);

  // Karaoke content rendering
  const getKaraokeContent = () => {
    const text = textRef.current;

    if (isSpeaking) {
      const spoken = text.slice(0, playbackPosition);

      if (!hasBeenPlayed) {
        // First time: show spoken + preview of next ~6 words
        const previewEnd = findWordsAhead(text, playbackPosition, 6);
        const preview = text.slice(playbackPosition, previewEnd);
        const hidden = text.slice(previewEnd);

        return (
          <>
            <span className="spoken">{spoken}</span>
            <span className="preview">{preview}</span>
            <span className="hidden">{hidden}</span>
          </>
        );
      }

      // Replay: show all with spoken highlighted
      const unspoken = text.slice(playbackPosition);
      return (
        <>
          <span className="spoken">{spoken}</span>
          <span className="unspoken">{unspoken}</span>
        </>
      );
    }

    // Not speaking
    if (hasBeenPlayed) {
      // Show full text after completion
      return <span className="spoken">{text}</span>;
    }

    // Never played - show first ~6 words as preview, rest hidden
    const previewEnd = findWordsAhead(text, 0, 6);
    return (
      <>
        <span className="preview">{text.slice(0, previewEnd)}</span>
        <span className="hidden">{text.slice(previewEnd)}</span>
      </>
    );
  };

  return (
    <div className="app">
      <h1>Pocket TTS Streaming Demo</h1>
      <p className="subtitle">Click the text to play/stop. Double-click to restart.</p>

      <div className="controls">
        <button onClick={handlePlay} disabled={isSpeaking}>
          Play
        </button>
        <button onClick={handleStop} disabled={!isSpeaking}>
          Stop
        </button>
        <label>
          Voice:
          <select value={voice} onChange={(e) => setVoice(e.target.value)}>
            <option value="cosette">Cosette</option>
            <option value="alba">Alba</option>
            <option value="brenda">Brenda</option>
          </select>
        </label>
      </div>

      <div
        className={`karaoke-text ${isSpeaking ? 'speaking' : ''}`}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
      >
        {getKaraokeContent()}
      </div>

      <div className="status">
        {isSpeaking ? 'Speaking...' : hasBeenPlayed ? 'Finished' : 'Ready'}
      </div>
    </div>
  );
}
