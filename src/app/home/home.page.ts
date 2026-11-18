import { DecimalPipe } from '@angular/common';
import {
  Component,
  ViewChild,
  inject,
  type ElementRef,
  type OnDestroy
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  IonButton,
  IonContent,
  IonIcon,
  IonRange
} from '@ionic/angular';
import { addIcons } from 'ionicons';
import {
  cameraReverseOutline,
  play,
  stop
} from 'ionicons/icons';

import type { Detection } from '../detector/detection.interface';
import { DetectorService } from '../detector/detector.service';


@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],

  imports: [
    DecimalPipe,
    FormsModule,
    IonContent,
    IonButton,
    IonIcon,
    IonRange
  ],
})

export class HomePage implements OnDestroy {

  @ViewChild('video')
  videoRef!: ElementRef<HTMLVideoElement>;

  @ViewChild('canvas')
  canvasRef!: ElementRef<HTMLCanvasElement>;


  // Estado de la aplicación
  running = false;
  loading = false;
  error = '';

  // FPS
  fps = 0;

  // Confianza mínima
  conf = 0.4;

  // Conteo por clase
  counts: {
    label: string;
    n: number;
  }[] = [];


  // Configuración de cámara
  private facing: 'environment' | 'user' = 'environment';

  private stream?: MediaStream;

  // Control del ciclo de detección
  private modelReady = false;
  private animationId = 0;

  // Cálculo de FPS
  private frames = 0;
  private lastTick = performance.now();

  // Servicio del detector
  private readonly detector = inject(DetectorService);


  constructor() {

    addIcons({
      cameraReverseOutline,
      play,
      stop
    });

  }


  /**
   * Iniciar o detener la detección.
   */
  async toggle(): Promise<void> {

    if (this.running) {

      this.stop();

    } else {

      await this.start();

    }

  }


  /**
   * Iniciar el modelo y la cámara.
   */
  async start(): Promise<void> {

    this.error = '';

    if (this.running || this.loading) {
      return;
    }

    try {

      if (!this.modelReady) {

        this.loading = true;

        await this.detector.load();

        this.modelReady = true;

        this.loading = false;

      }

      await this.openCamera();

      this.running = true;

      this.frames = 0;
      this.fps = 0;
      this.lastTick = performance.now();

      void this.loop();

    } catch (e: unknown) {

      this.loading = false;
      this.running = false;

      this.stream?.getTracks().forEach(
        track => track.stop()
      );

      this.stream = undefined;

      const msg = e instanceof Error
        ? e.message
        : String(e);

      this.error = 'No se pudo iniciar: ' + msg;

    }

  }


  /**
   * Detener cámara y detecciones.
   */
  stop(): void {

    this.running = false;

    cancelAnimationFrame(this.animationId);

    this.stream?.getTracks().forEach(
      track => track.stop()
    );

    this.stream = undefined;

    const video = this.videoRef?.nativeElement;

    if (video) {
      video.pause();
      video.srcObject = null;
    }

    this.clearCanvas();

    this.counts = [];
    this.fps = 0;

  }


  /**
   * Cambiar entre cámara frontal y trasera.
   */
  async switchCamera(): Promise<void> {

    this.facing =
      this.facing === 'environment'
        ? 'user'
        : 'environment';

    if (this.running) {

      try {

        await this.openCamera();

      } catch (e: unknown) {

        const msg = e instanceof Error
          ? e.message
          : String(e);

        this.error = 'Error cambiando cámara: ' + msg;

        this.stop();

      }

    }

  }


  /**
   * Solicitar acceso a la cámara.
   */
  private async openCamera(): Promise<void> {

    this.stream?.getTracks().forEach(
      track => track.stop()
    );

    this.stream = await navigator.mediaDevices.getUserMedia({

      video: {
        facingMode: this.facing,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },

      audio: false

    });

    const video = this.videoRef.nativeElement;

    video.srcObject = this.stream;

    await new Promise<void>((resolve, reject) => {

      video.onloadedmetadata = () => resolve();

      video.onerror = () => reject(
        new Error('No se pudo cargar el vídeo')
      );

    });

    await video.play();

    const canvas = this.canvasRef.nativeElement;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

  }


  /**
   * Ciclo de detección en tiempo real.
   */
  private async loop(): Promise<void> {

    if (!this.running) {
      return;
    }

    try {

      const video = this.videoRef.nativeElement;

      if (video.readyState >= 2) {

        const detections = await this.detector.detect(
          video,
          this.conf
        );

        // Evitar actualizar si se detuvo durante la inferencia
        if (!this.running) {
          return;
        }

        this.draw(detections);

        this.updateCounts(detections);

        this.updateFps();

      }

    } catch (e: unknown) {

      const msg = e instanceof Error
        ? e.message
        : String(e);

      this.error = 'Error en la detección: ' + msg;

      this.stop();

      return;

    }

    // Programar siguiente fotograma
    if (this.running) {

      this.animationId = requestAnimationFrame(
        () => void this.loop()
      );

    }

  }


  /**
   * Dibujar cajas delimitadoras.
   */
  private draw(dets: Detection[]): void {

    const canvas = this.canvasRef.nativeElement;

    const ctx = canvas.getContext('2d');

    if (!ctx) {
      return;
    }

    ctx.clearRect(
      0,
      0,
      canvas.width,
      canvas.height
    );

    ctx.lineWidth = Math.max(
      2,
      canvas.width / 320
    );

    ctx.font = `${Math.max(
      14,
      canvas.width / 45
    )}px sans-serif`;


    for (const d of dets) {

      const text = `${d.label} ${(d.score * 100).toFixed(0)}%`;

      const tw = ctx.measureText(text).width + 10;

      const th = parseInt(ctx.font, 10) + 8;


      // Caja delimitadora
      ctx.strokeStyle = '#00ff88';

      ctx.strokeRect(
        d.x,
        d.y,
        d.w,
        d.h
      );


      // Fondo de etiqueta
      const labelY = Math.max(0, d.y - th);

      ctx.fillStyle = '#00ff88';

      ctx.fillRect(
        d.x,
        labelY,
        tw,
        th
      );


      // Texto
      ctx.fillStyle = '#002b17';

      ctx.fillText(
        text,
        d.x + 5,
        Math.max(th - 6, d.y - 6)
      );

    }

  }


  /**
   * Actualizar conteo por clase.
   */
  private updateCounts(dets: Detection[]): void {

    const map = new Map<string, number>();

    dets.forEach(d => {

      map.set(
        d.label,
        (map.get(d.label) ?? 0) + 1
      );

    });

    this.counts = [...map].map(
      ([label, n]) => ({
        label,
        n
      })
    );

  }


  /**
   * Calcular fotogramas por segundo.
   */
  private updateFps(): void {

    this.frames++;

    const now = performance.now();

    if (now - this.lastTick >= 1000) {

      this.fps = this.frames;

      this.frames = 0;

      this.lastTick = now;

    }

  }


  /**
   * Limpiar Canvas.
   */
  private clearCanvas(): void {

    const canvas = this.canvasRef?.nativeElement;

    if (!canvas) {
      return;
    }

    const ctx = canvas.getContext('2d');

    ctx?.clearRect(
      0,
      0,
      canvas.width,
      canvas.height
    );

  }


  /**
   * Liberar recursos al salir de la página.
   */
  ngOnDestroy(): void {

    this.stop();

  }

}