/**
 * HTTP polling client for streaming text-to-speech.
 * No WebSocket required - uses POST requests with generation_id.
 */

export interface TTSConfig {
  voice?: string;
  temperature?: number;
  lsd_decode_steps?: number;
  noise_clamp?: number;
  eos_threshold?: number;
}

export interface PollingTTSOptions {
  onStart?: () => void;
  onProgress?: (elapsedSeconds: number, totalSeconds: number, charPosition: number) => void;
  onEnd?: (completed: boolean) => void;
  onError?: (error: string) => void;
}

/** Timing info for a text chunk, used for karaoke sync */
interface ChunkTiming {
  charStart: number;
  charEnd: number;
  audioStartTime: number;
  audioEndTime: number;
}

interface AudioChunkResponse {
  index: number;
  audio_base64: string;
  char_start: number;
  char_end: number;
  duration_ms: number;
}

interface PollResponse {
  chunks: AudioChunkResponse[];
  done: boolean;
  status: string;
  // BACKPRESSURE: Server could return these for throttling:
  // queued_text_chunks?: number;
  // queued_chars?: number;
}

export class PollingTTS {
  private generationId: string | null = null;
  private audioContext: AudioContext | null = null;
  private sampleRate = 24000;
  private nextPlayTime = 0;
  private playbackStartTime = 0;
  private isPlaying = false;
  private isPaused = false;
  private aborted = false;
  private options: PollingTTSOptions = {};

  // Karaoke sync
  private chunkTimings: ChunkTiming[] = [];
  private totalDuration = 0;
  private allAudioReceived = false;
  private progressInterval: number | null = null;

  // Polling state
  private pollTimer: number | null = null;
  private endSignaled = false;

  /**
   * Create a new TTS generation session.
   */
  async create(config: TTSConfig = {}): Promise<void> {
    this.stop();
    this.aborted = false;
    this.isPaused = false;
    this.endSignaled = false;
    this.totalDuration = 0;
    this.nextPlayTime = 0;
    this.playbackStartTime = 0;
    this.allAudioReceived = false;
    this.chunkTimings = [];

    const response = await fetch('/tts/polling/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        voice: config.voice || 'cosette',
        ...config,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to create generation: ${response.statusText}`);
    }

    const data = await response.json();
    this.generationId = data.generation_id;
    this.sampleRate = data.sample_rate || 24000;
    this.audioContext = new AudioContext({ sampleRate: this.sampleRate });

    this.options.onStart?.();
  }

  /**
   * Feed text to the generation.
   * Can be called multiple times as text streams in.
   */
  async feedText(text: string): Promise<void> {
    if (!this.generationId || this.aborted) return;
    if (this.endSignaled) {
      throw new Error('Cannot feed text after end() called');
    }

    const response = await fetch(`/tts/polling/${this.generationId}/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to feed text: ${error}`);
    }

    // BACKPRESSURE: Check queue depth and slow down if needed:
    // const data = await response.json();
    // if (data.queued_chars > 1000) {
    //   await new Promise(r => setTimeout(r, 100)); // Throttle
    // }
  }

  /**
   * Signal that no more text will be sent.
   * Start polling for audio and play as it arrives.
   */
  async end(): Promise<void> {
    if (!this.generationId || this.aborted || this.endSignaled) return;

    this.endSignaled = true;

    const response = await fetch(`/tts/polling/${this.generationId}/end`, {
      method: 'POST',
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to end generation: ${error}`);
    }

    // Start polling for audio
    this.startPolling();
  }

  /**
   * Convenience method: feed all text at once and start playback.
   */
  async speak(
    text: string,
    config: TTSConfig = {},
    options: PollingTTSOptions = {}
  ): Promise<void> {
    this.options = options;
    await this.create(config);
    await this.feedText(text);
    await this.end();
  }

  private startPolling() {
    if (this.pollTimer || this.aborted) return;
    this.poll();
  }

  private async poll() {
    if (!this.generationId || this.aborted) return;

    try {
      const response = await fetch(`/tts/polling/${this.generationId}/poll`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Poll failed: ${response.statusText}`);
      }

      const data: PollResponse = await response.json();

      // Schedule new audio chunks
      for (const chunk of data.chunks) {
        await this.scheduleChunk(chunk);
      }

      if (data.done) {
        // All audio received
        this.allAudioReceived = true;
        console.log('[PollingTTS] All audio received');
        // Don't stop yet - wait for audio to finish playing
        return;
      }

      // Continue polling (adaptive delay)
      const delay = data.chunks.length > 0 ? 30 : 80;
      this.pollTimer = window.setTimeout(() => {
        this.pollTimer = null;
        this.poll();
      }, delay);

    } catch (err) {
      console.error('[PollingTTS] Poll error:', err);
      this.options.onError?.(String(err));
    }
  }

  private async scheduleChunk(chunk: AudioChunkResponse) {
    if (!this.audioContext || this.aborted) return;

    // Decode base64 audio
    const binaryString = atob(chunk.audio_base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    // Convert int16 to float32
    const int16Array = new Int16Array(bytes.buffer);
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32Array[i] = int16Array[i] / 32768;
    }

    // Create audio buffer
    const audioBuffer = this.audioContext.createBuffer(
      1,
      float32Array.length,
      this.sampleRate
    );
    audioBuffer.getChannelData(0).set(float32Array);

    // Schedule for gapless playback
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);

    const startTime = Math.max(this.audioContext.currentTime, this.nextPlayTime);

    // Track when playback actually starts
    if (!this.isPlaying) {
      this.playbackStartTime = startTime;
      this.isPlaying = true;
      this.startProgressTracking();
    }

    source.start(startTime);
    this.nextPlayTime = startTime + audioBuffer.duration;
    this.totalDuration += audioBuffer.duration;

    // Store timing for karaoke sync
    this.chunkTimings.push({
      charStart: chunk.char_start,
      charEnd: chunk.char_end,
      audioStartTime: startTime,
      audioEndTime: this.nextPlayTime,
    });

    // Handle playback end
    const thisBufferEndTime = this.nextPlayTime;
    source.onended = () => {
      if (!this.audioContext || this.aborted) return;

      const currentTime = this.audioContext.currentTime;
      const isLastBuffer = thisBufferEndTime >= this.nextPlayTime - 0.01;
      const isPlaybackComplete = currentTime >= this.nextPlayTime - 0.05;

      if (this.allAudioReceived && isLastBuffer && isPlaybackComplete) {
        this.isPlaying = false;
        this.stopProgressTracking();
        this.options.onEnd?.(true);
      }
    };
  }

  private startProgressTracking() {
    if (this.progressInterval) return;

    this.progressInterval = window.setInterval(() => {
      if (!this.audioContext || !this.isPlaying || this.aborted) return;
      if (this.totalDuration <= 0) return;

      const currentTime = this.audioContext.currentTime;
      const elapsed = currentTime - this.playbackStartTime;

      if (elapsed >= 0 && elapsed <= this.totalDuration) {
        const charPosition = this.getCharacterPosition(currentTime);
        this.options.onProgress?.(elapsed, this.totalDuration, charPosition);
      }
    }, 50);
  }

  private getCharacterPosition(currentTime: number): number {
    if (this.chunkTimings.length === 0) {
      const elapsed = currentTime - this.playbackStartTime;
      return Math.floor(elapsed * 12); // Fallback estimate
    }

    for (const chunk of this.chunkTimings) {
      if (currentTime >= chunk.audioStartTime && currentTime < chunk.audioEndTime) {
        const chunkDuration = chunk.audioEndTime - chunk.audioStartTime;
        if (chunkDuration <= 0) return chunk.charStart;

        const progress = (currentTime - chunk.audioStartTime) / chunkDuration;
        const charRange = chunk.charEnd - chunk.charStart;
        return Math.floor(chunk.charStart + progress * charRange);
      }
    }

    // Before first chunk
    if (this.chunkTimings.length > 0 && currentTime < this.chunkTimings[0].audioStartTime) {
      return 0;
    }

    // After last chunk
    const lastChunk = this.chunkTimings[this.chunkTimings.length - 1];
    if (currentTime >= lastChunk.audioEndTime) {
      return lastChunk.charEnd;
    }

    return 0;
  }

  private stopProgressTracking() {
    if (this.progressInterval) {
      clearInterval(this.progressInterval);
      this.progressInterval = null;
    }
  }

  /**
   * Pause playback.
   */
  pause() {
    if (this.audioContext && this.isPlaying && !this.isPaused) {
      this.audioContext.suspend();
      this.isPaused = true;
    }
  }

  /**
   * Resume playback.
   */
  resume() {
    if (this.audioContext && this.isPaused) {
      this.audioContext.resume();
      this.isPaused = false;
    }
  }

  get paused(): boolean {
    return this.isPaused;
  }

  get active(): boolean {
    return this.isPlaying;
  }

  /**
   * Stop playback and cancel generation.
   */
  stop() {
    this.aborted = true;
    this.isPaused = false;
    this.stopProgressTracking();

    if (this.pollTimer) {
      clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    this.generationId = null;
    this.isPlaying = false;
  }
}
