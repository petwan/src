import { Point, Dataset } from './types';

let nextId = 0;
const getNextId = () => (nextId++).toString();

const generateUniformPoints = (count: number, cls: number, xRange: [number, number], yRange: [number, number], noise = 0): Point[] => {
  return Array.from({ length: count }, () => {
    const noiseX = noise > 0 ? (Math.random() - 0.5) * noise : 0;
    const noiseY = noise > 0 ? (Math.random() - 0.5) * noise : 0;
    return {
      id: getNextId(),
      x: Math.random() * (xRange[1] - xRange[0]) + xRange[0] + noiseX,
      y: Math.random() * (yRange[1] - yRange[0]) + yRange[0] + noiseY,
      class: cls,
    };
  });
};

export const generateDatasets = (): Record<string, Dataset> => {
  return {
    twoClassBasic: {
      name: "简单二分类",
      description: "两个明显分离的类别",
      points: [
        ...generateUniformPoints(30, 0, [0.1, 0.4], [0.1, 0.4]),
        ...generateUniformPoints(30, 1, [0.6, 0.9], [0.6, 0.9]),
      ],
    },
    multiClass: {
      name: "多分类",
      description: "四个类别，有一些重叠",
      points: [
        ...generateUniformPoints(20, 0, [0.1, 0.4], [0.1, 0.4]),
        ...generateUniformPoints(20, 1, [0.6, 0.9], [0.6, 0.9]),
        ...generateUniformPoints(20, 2, [0.1, 0.4], [0.6, 0.9]),
        ...generateUniformPoints(20, 3, [0.6, 0.9], [0.1, 0.4]),
      ],
    },
    concentricCircles: {
      name: "同心圆",
      description: "以同心圆形式排列的两个类别",
      points: Array.from({ length: 100 }, () => {
        const isInner = Math.random() > 0.5;
        const radius = isInner ? 0.1 + 0.1 * Math.random() : 0.3 + 0.1 * Math.random();
        const angle = Math.random() * Math.PI * 2;
        return {
          id: getNextId(),
          x: 0.5 + radius * Math.cos(angle),
          y: 0.5 + radius * Math.sin(angle),
          class: isInner ? 0 : 1,
        };
      }),
    },
    moonsDataset: {
      name: "月牙形",
      description: "两个月牙形状的类别（非线性边界）",
      points: Array.from({ length: 100 }, (_, i) => {
        const isTopMoon = i < 50;
        const angle = (i % 50) / 50 * Math.PI;
        const noise = (Math.random() - 0.5) * 0.05;
        
        let x, y;
        if (isTopMoon) {
          x = 0.3 + 0.2 * Math.cos(angle) + noise;
          y = 0.6 + 0.2 * Math.sin(angle) + noise;
        } else {
          x = 0.7 + 0.2 * Math.cos(angle + Math.PI) + noise;
          y = 0.4 + 0.2 * Math.sin(angle + Math.PI) + noise;
        }
        
        return {
          id: getNextId(),
          x: Math.min(Math.max(x, 0), 1),
          y: Math.min(Math.max(y, 0), 1),
          class: isTopMoon ? 0 : 1,
        };
      }),
    },
    xorProblem: {
      name: "异或问题",
      description: "非线性可分离模式（对于低k值的KNN较困难）",
      points: [
        ...generateUniformPoints(25, 0, [0.1, 0.4], [0.1, 0.4], 0.05),
        ...generateUniformPoints(25, 0, [0.6, 0.9], [0.6, 0.9], 0.05),
        ...generateUniformPoints(25, 1, [0.1, 0.4], [0.6, 0.9], 0.05),
        ...generateUniformPoints(25, 1, [0.6, 0.9], [0.1, 0.4], 0.05),
      ],
    },
  };
};

export const classColors: Record<number, string> = {
  0: '#3b82f6', // blue
  1: '#ef4444', // red
  2: '#10b981', // green
  3: '#f59e0b', // yellow
  4: '#8b5cf6', // purple
};