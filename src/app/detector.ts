import { Service } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

export interface Detection {
  x: number; y: number; w: number; h: number; // en píxeles del video original
  label: string;
  score: number; // 0 a 1
}

const SIZE = 640; // mismo imgsz con el que exportaste
const IOU = 0.5;  // umbral de NMS
const LABELS = ['clase0', 'clase1']; // <- tus clases, en el orden del entrenamiento

@Service()
export class Detector {
  private model!: tf.GraphModel;

  async load(): Promise<void> {
    await tf.setBackend('webgl');
    await tf.ready();
    this.model = await tf.loadGraphModel('assets/model/model.json');
  }

  async detect(video: HTMLVideoElement, conf: number): Promise<Detection[]> {
    const nc = LABELS.length;

    const [boxes, scores, classes] = tf.tidy(() => {
      const img = tf.browser.fromPixels(video)
        .resizeBilinear([SIZE, SIZE]).div(255).expandDims(0);
      // Salida YOLOv8: [1, 4+nc, 8400] -> [8400, 4+nc]
      const out = (this.model.execute(img) as tf.Tensor).squeeze([0]).transpose([1, 0]);
      const [xywh, cls] = tf.split(out, [4, nc], 1);
      const cx = xywh.slice([0, 0], [-1, 1]);
      const cy = xywh.slice([0, 1], [-1, 1]);
      const w = xywh.slice([0, 2], [-1, 1]);
      const h = xywh.slice([0, 3], [-1, 1]);
      const b = tf.concat([
        cy.sub(h.div(2)), cx.sub(w.div(2)),
        cy.add(h.div(2)), cx.add(w.div(2)),
      ], 1);
      return [b, cls.max(1), cls.argMax(1)] as tf.Tensor[];
    });

    const keep = await tf.image.nonMaxSuppressionAsync(
      boxes as tf.Tensor2D, scores as tf.Tensor1D, 50, IOU, conf);

    const idx = await keep.data();
    const b = (await boxes.array()) as number[][];
    const s = await scores.data();
    const c = await classes.data();
    tf.dispose([boxes, scores, classes, keep]);

    const sx = video.videoWidth / SIZE;
    const sy = video.videoHeight / SIZE;

    return Array.from(idx).map(i => {
      const [y1, x1, y2, x2] = b[i];
      return {
        x: x1 * sx,
        y: y1 * sy,
        w: (x2 - x1) * sx,
        h: (y2 - y1) * sy,
        label: LABELS[c[i]] ?? `clase ${c[i]}`,
        score: s[i],
      };
    });
  }
}