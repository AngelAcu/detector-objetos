
import { Injectable } from '@angular/core';
import * as ort from 'onnxruntime-web';

import type { Detection } from './detection.interface';

@Injectable({
    providedIn: 'root'
})
export class DetectorService {

    private session?: ort.InferenceSession;

    // Configuración del modelo
    private readonly MODEL_PATH = 'assets/models/best.onnx';

    private readonly INPUT_SIZE = 640;

    private readonly CLASS_NAMES = [
        'mobile-phone',
        'mouse'
    ];

    // Umbral de IoU para NMS
    private readonly NMS_THRESHOLD = 0.45;

    // Límite de detecciones
    private readonly MAX_DETECTIONS = 100;

    /**
     * Carga el modelo ONNX.
     */
    async load(): Promise<void> {

        if (this.session) {
            return;
        }

        try {

            // Utilizar WebAssembly para ejecutar el modelo
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.proxy = false;

            this.session = await ort.InferenceSession.create(
                this.MODEL_PATH,
                {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all'
                }
            );

            console.log('Modelo YOLO cargado correctamente');

            console.log(
                'Entradas:',
                this.session.inputNames
            );

            console.log(
                'Salidas:',
                this.session.outputNames
            );

        } catch (error: unknown) {

            this.session = undefined;

            console.error('Error cargando modelo:', error);

            throw new Error(
                'No se pudo cargar el modelo ONNX',
                { cause: error }
            );
        }
    }

    /**
     * Detecta objetos en un fotograma de vídeo.
     */
    async detect(
        video: HTMLVideoElement,
        confidence: number
    ): Promise<Detection[]> {

        if (!this.session) {
            throw new Error('El modelo no está cargado');
        }

        if (
            video.videoWidth === 0 ||
            video.videoHeight === 0
        ) {
            return [];
        }

        // Preparar imagen
        const prepared = this.preprocess(video);

        // Crear tensor de entrada
        const tensor = new ort.Tensor(
            'float32',
            prepared.data,
            [1, 3, this.INPUT_SIZE, this.INPUT_SIZE]
        );

        // Ejecutar inferencia
        const results = await this.session.run({
            [this.session.inputNames[0]]: tensor
        });

        // Obtener salida del modelo
        const output = results[
            this.session.outputNames[0]
        ];

        // Interpretar predicciones
        return this.postprocess(
            output,
            confidence,
            prepared.scale,
            prepared.padX,
            prepared.padY,
            video.videoWidth,
            video.videoHeight
        );
    }

    /**
     * Redimensiona la imagen manteniendo proporciones.
     * Utiliza Letterbox para completar a 640x640.
     */
    private preprocess(video: HTMLVideoElement) {

        const size = this.INPUT_SIZE;

        const originalWidth = video.videoWidth;
        const originalHeight = video.videoHeight;

        const scale = Math.min(
            size / originalWidth,
            size / originalHeight
        );

        const newWidth = Math.round(originalWidth * scale);
        const newHeight = Math.round(originalHeight * scale);

        const padX = (size - newWidth) / 2;
        const padY = (size - newHeight) / 2;

        const canvas = document.createElement('canvas');

        canvas.width = size;
        canvas.height = size;

        const ctx = canvas.getContext('2d', {
            willReadFrequently: true
        });

        if (!ctx) {
            throw new Error('No se pudo crear el contexto Canvas');
        }

        // Color de relleno utilizado por YOLO
        ctx.fillStyle = 'rgb(114, 114, 114)';

        ctx.fillRect(0, 0, size, size);

        // Dibujar imagen redimensionada
        ctx.drawImage(
            video,
            0,
            0,
            originalWidth,
            originalHeight,
            padX,
            padY,
            newWidth,
            newHeight
        );

        const image = ctx.getImageData(
            0,
            0,
            size,
            size
        );

        const pixels = image.data;

        const area = size * size;

        // Tensor en formato NCHW: RGB
        const data = new Float32Array(3 * area);

        for (let i = 0; i < area; i++) {

            const pixel = i * 4;

            data[i] = pixels[pixel] / 255;

            data[area + i] = pixels[pixel + 1] / 255;

            data[2 * area + i] = pixels[pixel + 2] / 255;
        }

        return {
            data,
            scale,
            padX,
            padY
        };
    }

    /**
     * Procesa las predicciones del modelo YOLO.
     */
    private postprocess(
        output: ort.Tensor,
        confidence: number,
        scale: number,
        padX: number,
        padY: number,
        originalWidth: number,
        originalHeight: number
    ): Detection[] {

        const dims = output.dims;

        const data = output.data as Float32Array;

        // Para este modelo esperamos 4 coordenadas
        // y 2 puntuaciones de clase.
        const features = 4 + this.CLASS_NAMES.length;

        let numPredictions: number;

        let channelsFirst: boolean;

        if (dims.length !== 3 || dims[0] !== 1) {
            throw new Error(
                `Forma de salida YOLO no compatible: ${dims.join('x')}`
            );
        }

        if (dims[1] === features) {

            // Formato [1, 6, N]
            channelsFirst = true;
            numPredictions = dims[2];

        } else if (dims[2] === features) {

            // Formato [1, N, 6]
            channelsFirst = false;
            numPredictions = dims[1];

        } else {

            throw new Error(
                `Salida inesperada: ${dims.join('x')}. ` +
                `Se esperaban 6 características por predicción.`
            );
        }

        const candidates: Detection[] = [];

        for (let i = 0; i < numPredictions; i++) {

            const getValue = (channel: number): number => {

                return channelsFirst
                    ? data[channel * numPredictions + i]
                    : data[i * features + channel];

            };

            // Coordenadas del centro y dimensiones
            const cx = getValue(0);
            const cy = getValue(1);
            const width = getValue(2);
            const height = getValue(3);

            // Buscar la clase con mayor confianza
            let maxScore = -Infinity;
            let classId = -1;

            for (
                let c = 0;
                c < this.CLASS_NAMES.length;
                c++
            ) {

                const score = getValue(4 + c);

                if (score > maxScore) {
                    maxScore = score;
                    classId = c;
                }
            }

            if (
                !Number.isFinite(maxScore) ||
                !Number.isFinite(cx) ||
                !Number.isFinite(cy) ||
                !Number.isFinite(width) ||
                !Number.isFinite(height)
            ) {
                continue;
            }

            if (maxScore < confidence) {
                continue;
            }

            // YOLO devuelve centro, ancho y alto.
            // Convertimos a esquinas.
            const x1 = (cx - width / 2 - padX) / scale;
            const y1 = (cy - height / 2 - padY) / scale;

            const x2 = (cx + width / 2 - padX) / scale;
            const y2 = (cy + height / 2 - padY) / scale;

            // Limitar coordenadas al vídeo original
            const left = Math.max(
                0,
                Math.min(originalWidth, x1)
            );

            const top = Math.max(
                0,
                Math.min(originalHeight, y1)
            );

            const right = Math.max(
                0,
                Math.min(originalWidth, x2)
            );

            const bottom = Math.max(
                0,
                Math.min(originalHeight, y2)
            );

            const boxWidth = right - left;
            const boxHeight = bottom - top;

            if (boxWidth <= 0 || boxHeight <= 0) {
                continue;
            }

            candidates.push({
                label: this.CLASS_NAMES[classId],
                classId,
                score: maxScore,
                x: left,
                y: top,
                w: boxWidth,
                h: boxHeight
            });
        }

        // Eliminar cajas duplicadas
        return this.nms(
            candidates,
            this.NMS_THRESHOLD
        );
    }

    /**
     * Non-Maximum Suppression.
     * Elimina detecciones superpuestas de la misma clase.
     */
    private nms(
        detections: Detection[],
        threshold: number
    ): Detection[] {

        const sorted = [...detections].sort(
            (a, b) => b.score - a.score
        );

        const selected: Detection[] = [];

        while (
            sorted.length > 0 &&
            selected.length < this.MAX_DETECTIONS
        ) {

            const best = sorted.shift()!;

            selected.push(best);

            for (let i = sorted.length - 1; i >= 0; i--) {

                const candidate = sorted[i];

                // No comparar clases diferentes
                if (best.classId !== candidate.classId) {
                    continue;
                }

                const iou = this.calculateIoU(
                    best,
                    candidate
                );

                if (iou > threshold) {
                    sorted.splice(i, 1);
                }
            }
        }

        return selected;
    }

    /**
     * Calcula Intersection over Union.
     */
    private calculateIoU(
        a: Detection,
        b: Detection
    ): number {

        const x1 = Math.max(a.x, b.x);
        const y1 = Math.max(a.y, b.y);

        const x2 = Math.min(
            a.x + a.w,
            b.x + b.w
        );

        const y2 = Math.min(
            a.y + a.h,
            b.y + b.h
        );

        const intersectionWidth = Math.max(
            0,
            x2 - x1
        );

        const intersectionHeight = Math.max(
            0,
            y2 - y1
        );

        const intersection =
            intersectionWidth * intersectionHeight;

        const areaA = a.w * a.h;
        const areaB = b.w * b.h;

        const union = areaA + areaB - intersection;

        return union > 0
            ? intersection / union
            : 0;
    }

}