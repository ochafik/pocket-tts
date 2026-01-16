import React, { useState, useCallback, useRef, useEffect } from 'react';
import { PollingTTS } from './polling-tts';

const LOREM_IPSUM = `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit, nec luctus magna felis sollicitudin mauris. Integer in mauris eu nibh euismod gravida.

Duis ac tellus et risus vulputate vehicula. Donec lobortis risus a elit. Etiam tempor ultrices risus. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.`;

/**
 * Find word boundary for clean highlighting.
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
    while (pos < text.length && text[pos] !== ' ') pos++;
    while (pos < text.length && text[pos] === ' ') pos++;
    wordsFound++;
  }
  return pos;
}

export function App() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [playbackPosition, setPlaybackPosition] = useState(0);
  const [hasBeenPlayed, setHasBeenPlayed] = useState(false);
  const [voice, setVoice] = useState('cosette');

  const ttsRef = useRef<PollingTTS | null>(null);
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

    const tts = new PollingTTS();
    ttsRef.current = tts;
    lastBoundaryRef.current = -1;

    setIsSpeaking(true);
    setIsPaused(false);
    setPlaybackPosition(0);

    tts.speak(textRef.current, { voice }, {
      onStart: () => {
        console.log('[Polling TTS] Started');
      },
      onProgress: (_elapsed, _total, charPosition) => {
        const wordBoundary = findWordBoundary(textRef.current, charPosition);
        const advancedBoundary = findWordsAhead(textRef.current, wordBoundary, 1);
        const clampedBoundary = Math.min(advancedBoundary, textRef.current.length);

        if (clampedBoundary !== lastBoundaryRef.current) {
          lastBoundaryRef.current = clampedBoundary;
          setPlaybackPosition(clampedBoundary);
        }
      },
      onEnd: (completed) => {
        console.log('[Polling TTS] Ended, completed:', completed);
        setIsSpeaking(false);
        if (completed) {
          setPlaybackPosition(textRef.current.length);
          setHasBeenPlayed(true);
        }
      },
      onError: (error) => {
        console.error('[Polling TTS] Error:', error);
        setIsSpeaking(false);
      },
    }).catch((err) => {
      console.error('[Polling TTS] Start error:', err);
      setIsSpeaking(false);
    });
  }, [isSpeaking, voice]);

  const handleStop = useCallback(() => {
    ttsRef.current?.stop();
    setIsSpeaking(false);
    setIsPaused(false);
  }, []);

  const handlePauseResume = useCallback(() => {
    const tts = ttsRef.current;
    if (!tts) {
      handlePlay();
      return;
    }

    if (tts.paused) {
      tts.resume();
      setIsPaused(false);
    } else if (tts.active) {
      tts.pause();
      setIsPaused(true);
    } else {
      handlePlay();
    }
  }, [handlePlay]);

  const handleClick = useCallback(() => {
    handlePauseResume();
  }, [handlePauseResume]);

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setPlaybackPosition(0);
    setHasBeenPlayed(false);
    handleStop();
    setTimeout(handlePlay, 100);
  }, [handlePlay, handleStop]);

  const getKaraokeContent = () => {
    const text = textRef.current;

    if (isSpeaking) {
      const spoken = text.slice(0, playbackPosition);

      if (!hasBeenPlayed) {
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

      const unspoken = text.slice(playbackPosition);
      return (
        <>
          <span className="spoken">{spoken}</span>
          <span className="unspoken">{unspoken}</span>
        </>
      );
    }

    if (hasBeenPlayed) {
      return <span className="spoken">{text}</span>;
    }

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
      <h1>Pocket TTS Polling Demo</h1>
      <p className="subtitle">
        HTTP polling instead of WebSocket. Click text to play/pause, double-click to restart.
      </p>

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
        {isPaused ? 'Paused' : isSpeaking ? 'Speaking...' : hasBeenPlayed ? 'Finished' : 'Ready'}
      </div>

      <div className="info">
        <strong>How it works:</strong> POST /tts/polling/create to start, POST .../text to feed text,
        POST .../end to signal completion, POST .../poll to get audio chunks.
      </div>
    </div>
  );
}
