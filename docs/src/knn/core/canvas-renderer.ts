
import { Point, TestPoint } from './types'
import { KNNEngine } from './knn-engine'

/**
 * Canvas 渲染器：负责 KNN 可视化的所有绘制逻辑
 * - 支持动画、决策边界、置信度环、邻居连线等
 * - 所有坐标在 [0,1] 归一化空间中计算，渲染时映射到像素
 */

export class CanvasRenderer {
    private ctx: CanvasRenderingContext2D;
    private animationId: number | null = null;
    private time: number = 0; // 动画时间戳（用于 sine 波动等效果）

    constructor(canvas: HTMLCanvasElement) {
        this.ctx = canvas.getContext('2d')!;
    }

    startAnimation() {
        this.stopAnimation();
        const animate = () => {
        this.time++;
        this.animationId = requestAnimationFrame(animate);
        };
        this.animationId = requestAnimationFrame(animate);
    }

    stopAnimation() {
        if (this.animationId) {
        cancelAnimationFrame(this.animationId);
        this.animationId = null;
        }
    }

    clearCanvas(width: number, height: number) {
        this.ctx.clearRect(0, 0, width, height);
    }

    drawGrid(width: number, height: number, gridSize = 40) {
        this.ctx.strokeStyle = '#EEEEEE';
        this.ctx.lineWidth = 1;
        
        // 垂直线
        for (let x = 0; x <= width; x += gridSize) {
          this.ctx.beginPath();
          this.ctx.moveTo(x, 0);
          this.ctx.lineTo(x, height);
          this.ctx.stroke();
        }

        // 水平线
        for (let y = 0; y <= height; y += gridSize) {
          this.ctx.beginPath();
          this.ctx.moveTo(0, y);
          this.ctx.lineTo(width, y);
          this.ctx.stroke();
        }
    }

    drawPoint(x: number, y: number, color: string, radius = 5, isSelected = false) {
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
        
        if (isSelected) {
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        }
    }

    drawLine(x1: number, y1: number, x2: number, y2: number, color = '#000000', width = 1, dash: number[] = []) {
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = width;
        
        if (dash.length) {
        this.ctx.setLineDash(dash);
        } else {
        this.ctx.setLineDash([]);
        }
        
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }

    drawConfidenceRing(x: number, y: number, confidence: number, color: string, time: number, enhancedMode: boolean) {
        const percentage = Math.round(100 * confidence);
        
        // 绘制百分比文本
        this.ctx.font = '10px Arial';
        this.ctx.fillStyle = 'white';
        this.ctx.strokeStyle = 'black';
        this.ctx.lineWidth = 1;
        this.ctx.textAlign = 'center';
        this.ctx.strokeText(`${percentage}%`, x, y - 15);
        this.ctx.fillText(`${percentage}%`, x, y - 15);
        
        // 绘制置信度圆弧
        const angle = 2 * Math.PI * confidence;
        const radius = 12;
        const animationFactor = enhancedMode ? 0.5 + 0.5 * Math.sin(0.005 * time) : 1;
        
        this.ctx.save();
        this.ctx.shadowColor = color;
        this.ctx.shadowBlur = 5 * animationFactor;
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, -Math.PI / 2, -Math.PI / 2 + angle);
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        this.ctx.restore();
        
        // 绘制完整圆作为背景
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.strokeStyle = 'rgba(200, 200, 200, 0.3)';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
    }

    // 添加绘制决策边界的方法
    drawDecisionBoundary(
        trainingPoints: Point[],
        width: number,
        height: number,
        k: number,
        distanceMetric: 'euclidean' | 'manhattan',
        useWeightedKNN: boolean = false,
        classColors: Record<number, string>
      ) {
        if (trainingPoints.length === 0) return;

        // 创建临时 KNNEngine
        const knn = new KNNEngine();
        knn.setTrainingPoints(trainingPoints);

        // 绘制决策边界网格
        const gridSize = 10;
        const padding = 0;

        // 保存当前绘图状态
        this.ctx.save();
        this.ctx.globalAlpha = 1.0; // 降低透明度以确保网格线可见

        for (let x = padding; x <= width - padding; x += gridSize) {
            for (let y = padding; y <= height - padding; y += gridSize) {
            // 归一化坐标
            const normalizedX = (x - padding) / (width - 2 * padding);
            const normalizedY = (y - padding) / (height - 2 * padding);

            const testPoint: Point = {
                id: 'boundary',
                x: Math.max(0, Math.min(1, normalizedX)),
                y: Math.max(0, Math.min(1, normalizedY)),
                class: -1,
            };

            // ✅ 直接使用真实引擎分类
            const result = knn.classifyPoint(testPoint, k, distanceMetric, useWeightedKNN);
            
            if (result.predictedClass !== -1) {
                const color = classColors[result.predictedClass] || '#CCCCCC';
                // 将十六进制颜色转换为RGBA格式，以便更好地控制透明度
                const r = parseInt(color.slice(1, 3), 16);
                const g = parseInt(color.slice(3, 5), 16);
                const b = parseInt(color.slice(5, 7), 16);
                this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.1)`;
                this.ctx.fillRect(x - gridSize / 2, y - gridSize / 2, gridSize, gridSize);
            }
            }
        }

        // 恢复绘图状态
        this.ctx.restore();
    }

    // mapToCanvas & mapFromCanvas
    mapToCanvas(point: Point, width: number, height: number, padding = 20) {
        return {
        x: padding + point.x * (width - 2 * padding),
        y: padding + point.y * (height - 2 * padding),
        };
    }
    
    mapFromCanvas(canvasX: number, canvasY: number, width: number, height: number, padding = 20): Point {
        return {
        id: 'temp',
        x: Math.max(0, Math.min(1, (canvasX - padding) / (width - 2 * padding))),
        y: Math.max(0, Math.min(1, (canvasY - padding) / (height - 2 * padding))),
        class: -1,
        };
    }

    render(
        trainingPoints: Point[],
        testPoints: TestPoint[],
        config: {
          showDecisionBoundary: boolean;
          showConfidence: boolean;
          animateDistances: boolean;
          enhancedMode: boolean;
          classColors: Record<number, string>;
          useWeightedKNN: boolean;
          k: number;
          distanceMetric: 'euclidean' | 'manhattan';
        }
      ) {
        const width = this.ctx.canvas.width;
        const height = this.ctx.canvas.height;
        
        this.clearCanvas(width, height);
        
        // 先绘制网格线
        this.drawGrid(width, height);
        
        // 绘制决策边界（如果启用）
        if (config.showDecisionBoundary) {
            this.drawDecisionBoundary(
                trainingPoints,
                width,
                height,
                config.k,
                config.distanceMetric,
                config.useWeightedKNN,
                config.classColors
            );
        }
        
        // 绘制训练点
        trainingPoints.forEach(point => {
        const canvasPos = this.mapToCanvas(point, width, height);
        const color = config.classColors[point.class] || '#6B7280';
        this.drawPoint(canvasPos.x, canvasPos.y, color);
        });
        
        // 绘制测试点
        testPoints.forEach((testPoint, pointIndex) => {
        const canvasPos = this.mapToCanvas(testPoint, width, height);
        
        // 绘制到邻居的连接线
        if (testPoint.nearestNeighbors) {
            testPoint.nearestNeighbors.forEach((neighbor, idx) => {
            const neighborPos = this.mapToCanvas(neighbor, width, height);
            
            if (config.animateDistances) {
                const progress = Math.min(1, Math.max(0, (this.time - (200 * pointIndex + 100 * idx)) / 1000));
                
                if (progress > 0) {
                const intermediateX = neighborPos.x + (canvasPos.x - neighborPos.x) * (1 - progress);
                const intermediateY = neighborPos.y + (canvasPos.y - neighborPos.y) * (1 - progress);
                
                this.drawLine(
                    neighborPos.x,
                    neighborPos.y,
                    intermediateX,
                    intermediateY,
                    'rgba(0, 0, 0, 0.3)',
                    1,
                    [5, 2]
                );
                
                // 绘制邻近点带动画效果
                const pulseSize = config.enhancedMode ? 1 + 0.3 * Math.sin(0.003 * this.time + idx) : 1;
                this.drawPoint(neighborPos.x, neighborPos.y, config.classColors[neighbor.class] || '#6B7280', 5 * pulseSize, true);
                }
            } else {
                this.drawLine(
                canvasPos.x,
                canvasPos.y,
                neighborPos.x,
                neighborPos.y,
                'rgba(0, 0, 0, 0.3)',
                1,
                [5, 2]
                );
                this.drawPoint(neighborPos.x, neighborPos.y, config.classColors[neighbor.class] || '#6B7280', 5, true);
            }
            });
        }
        
        // 绘制测试点
        const predictedColor = testPoint.predictedClass !== undefined ? 
            config.classColors[testPoint.predictedClass] : '#999999';
        const size = config.enhancedMode ? 7 * (1 + 0.2 * Math.sin(0.005 * this.time)) : 7;
        this.drawPoint(canvasPos.x, canvasPos.y, predictedColor, size, true);
        
        // 绘制置信度
        if (config.showConfidence && testPoint.confidence !== undefined) {
            this.drawConfidenceRing(canvasPos.x, canvasPos.y, testPoint.confidence, predictedColor, this.time, config.enhancedMode);
        }
        });
      }
    }