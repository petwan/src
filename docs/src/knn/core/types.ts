export interface Point {
    id: string;
    x: number;
    y: number;
    class: number;
}

export interface TestPoint extends Point {
    predictedClass?: number;
    confidence?: number;
    nearestNeighbors?: Point[];
}

export interface Dataset {
    name: string;
    description: string;
    points: Point[];
}

export interface KNNConfig {
    k: number;
    distanceMetric: 'euclidean' | 'manhattan';
    useWeightedKNN: boolean;
    showConfidence: boolean;
    showDecisionBoundary: boolean;
    animateDistances: boolean;
    enhancedMode: boolean;
    selectedDataset: string;
    gameMode?: boolean;
}