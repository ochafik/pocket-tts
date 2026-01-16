/**
 * WebSocket client for streaming text-to-speech.
 * Simplified design based on voice-assist's streamingWavPlayer.
 */

export interface TTSConfig {
  voice?: string;
  temperature?: number;
  lsd_decode_steps?: number;
  noise_clamp?: number;
  eos_threshold?: number;
}

export interface StreamingTTSOptions {
  onStart?: () => void;
  onProgress?: (elapsedSeconds: number, totalSeconds: number) => void;
  onEnd?: (completed: boolean) => void;
  onError?: (error: string) => void;
}

export class StreamingTTS {
  private ws: WebSocket | null = null;
  private audioContext: AudioContext | null = null;
  private sampleRate = 24000;
  private nextPlayTime = 0;
  private playbackStartTime = 0;
  private totalDuration = 0;
  private finalDuration = 0; // Set once all audio received
  private isPlaying = false;
  private options: StreamingTTSOptions = {};
  private progressInterval: number | null = null;
  private aborted = false;
  private allAudioReceived = false;

  /**
   * Start streaming TTS session.
   */
  async start(
    text: string,
    config: TTSConfig = {},
    options: StreamingTTSOptions = {}
  ): Promise<void> {
    this.stop();
    this.options = options;
    this.aborted = false;
    this.totalDuration = 0;
    this.finalDuration = 0;
    this.nextPlayTime = 0;
    this.playbackStartTime = 0;
    this.allAudioReceived = false;

    return new Promise((resolve, reject) => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/tts/stream`;

      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        // Send start message
        this.ws!.send(JSON.stringify({
          type: 'start',
          voice: config.voice || 'cosette',
          ...config,
        }));
      };

      this.ws.onmessage = async (event) => {
        if (this.aborted) return;

        if (event.data instanceof Blob) {
          await this.handleAudioData(event.data);
        } else {
          const msg = JSON.parse(event.data);
          this.handleControlMessage(msg, text, resolve, reject);
        }
      };

      this.ws.onerror = () => {
        this.options.onError?.('WebSocket error');
        reject(new Error('WebSocket error'));
      };

      this.ws.onclose = () => {
        // Don't stop progress tracking here - audio may still be playing!
        // Progress tracking is stopped in source.onended when the last buffer finishes
      };
    });
  }

  private handleControlMessage(
    msg: { type: string; [key: string]: unknown },
    text: string,
    resolve: () => void,
    reject: (err: Error) => void
  ) {
    switch (msg.type) {
      case 'started':
        this.sampleRate = (msg.sample_rate as number) || 24000;
        this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
        this.playbackStartTime = 0;

        // Send the full text
        this.ws?.send(JSON.stringify({ type: 'text', content: text }));
        this.ws?.send(JSON.stringify({ type: 'end' }));

        this.options.onStart?.();
        resolve();
        break;

      case 'done':
        // All audio received - now we have stable duration for progress
        this.allAudioReceived = true;
        this.finalDuration = this.totalDuration;
        break;

      case 'error':
        this.options.onError?.(msg.message as string);
        reject(new Error(msg.message as string));
        break;
    }
  }

  private async handleAudioData(blob: Blob) {
    if (!this.audioContext || this.aborted) return;

    const arrayBuffer = await blob.arrayBuffer();
    const int16Array = new Int16Array(arrayBuffer);

    // Convert int16 to float32
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

    // Handle playback end - only stop when ALL audio has finished
    // Capture the scheduled end time for THIS specific buffer at creation time
    const thisBufferEndTime = this.nextPlayTime;

    source.onended = () => {
      if (!this.audioContext || this.aborted) return;

      const currentTime = this.audioContext.currentTime;
      // Check if THIS buffer was the last one by comparing its end time to the final nextPlayTime
      const isLastBuffer = thisBufferEndTime >= this.nextPlayTime - 0.01;
      const isPlaybackComplete = currentTime >= this.nextPlayTime - 0.05;

      // Only stop when: all audio received AND this was the last scheduled buffer AND playback is complete
      if (this.allAudioReceived && isLastBuffer && isPlaybackComplete) {
        this.isPlaying = false;
        this.stopProgressTracking();
        this.options.onEnd?.(true);
      }
    };
  }

  private startProgressTracking() {
    if (this.progressInterval) return;

    // Update progress every 50ms (same as voice-assist)
    this.progressInterval = window.setInterval(() => {
      if (!this.audioContext || !this.isPlaying || this.aborted) return;
      if (this.totalDuration <= 0) return;

      const currentTime = this.audioContext.currentTime;
      const elapsed = currentTime - this.playbackStartTime;

      // Use finalDuration once all audio received, otherwise use growing totalDuration
      const duration = this.allAudioReceived ? this.finalDuration : this.totalDuration;

      if (elapsed >= 0 && elapsed <= duration) {
        this.options.onProgress?.(elapsed, duration);
      }
    }, 50);
  }

  private stopProgressTracking() {
    if (this.progressInterval) {
      clearInterval(this.progressInterval);
      this.progressInterval = null;
    }
  }

  /**
   * Stop playback.
   */
  stop() {
    this.aborted = true;
    this.stopProgressTracking();
    this.ws?.close();
    this.ws = null;

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    this.isPlaying = false;
  }
}
