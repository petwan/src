import { Point, TestPoint } from './types';

export class KNNEngine {
  private trainingPoints: Point[] = [];
  
  setTrainingPoints(points: Point[]) {
    this.trainingPoints = points;
  }
  
  euclideanDistance(p1: Point, p2: Point): number {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    return Math.sqrt(dx * dx + dy * dy);
  }
  
  manhattanDistance(p1: Point, p2: Point): number {
    return Math.abs(p2.x - p1.x) + Math.abs(p2.y - p1.y);
  }
  
  calculateDistance(p1: Point, p2: Point, metric: 'euclidean' | 'manhattan'): number {
    return metric === 'manhattan' ? this.manhattanDistance(p1, p2) : this.euclideanDistance(p1, p2);
  }
  
  classifyPoint(
    testPoint: Point,
    k: number,
    distanceMetric: 'euclidean' | 'manhattan',
    useWeightedKNN: boolean
  ): { predictedClass: number; confidence: number; neighbors: Point[] } {
    if (this.trainingPoints.length === 0) {
      return { predictedClass: -1, confidence: 0, neighbors: [] };
    }
    
    // 计算到所有训练点的距离
    const distances = this.trainingPoints.map(trainPoint => ({
      point: trainPoint,
      distance: this.calculateDistance(testPoint, trainPoint, distanceMetric),
    }));
    
    // 按距离排序
    distances.sort((a, b) => a.distance - b.distance);
    
    // 取k个最近邻
    const kNearest = distances.slice(0, k);
    const neighbors = kNearest.map(item => item.point);
    
    // 计算类别投票
    const votes: Record<number, number> = {};
    
    if (useWeightedKNN) {
      // 加权投票
      kNearest.forEach(item => {
        const { point, distance } = item;
        if (!votes[point.class]) votes[point.class] = 0;
        votes[point.class] += 1 / (distance + 1e-5);
      });
    } else {
      // 简单投票
      kNearest.forEach(item => {
        const { point } = item;
        if (!votes[point.class]) votes[point.class] = 0;
        votes[point.class]++;
      });
    }
    
    // 找出得票最多的类别
    let predictedClass = -1;
    let maxVotes = 0;
    Object.entries(votes).forEach(([cls, voteCount]) => {
      if (voteCount > maxVotes) {
        predictedClass = parseInt(cls);
        maxVotes = voteCount;
      }
    });
    
    // 计算置信度
    const totalVotes = Object.values(votes).reduce((sum, val) => sum + val, 0);
    const confidence = totalVotes > 0 ? maxVotes / totalVotes : 0;
    
    return {
      predictedClass,
      confidence,
      neighbors,
    };
  }
}
