export interface Detection {
  // Nombre de la clase detectada
  label: string;

  // Identificador de la clase
  classId: number;

  // Porcentaje de confianza (0 a 1)
  score: number;

  // Coordenadas de la caja delimitadora
  x: number;
  y: number;
  w: number;
  h: number;
}
