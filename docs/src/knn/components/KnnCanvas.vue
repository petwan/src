<template>
    <div class="canvas-container">
        <canvas ref="canvasRef" :width="canvasSize.width" :height="canvasSize.height"
            @pointerdown="handlePointerDown" />
    </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { Point, TestPoint, KNNConfig } from '../core/types'
import { classColors } from '../core/datasets'
import { CanvasRenderer } from '../core/canvas-renderer'

interface Props {
    trainingPoints: Point[];
    testPoints: TestPoint[];
    config: KNNConfig;
}

const props = defineProps<Props>();
const emit = defineEmits<{
    'point-added': [point: any];
}>();

const canvasRef = ref<HTMLCanvasElement | null>(null);
const renderer = ref<CanvasRenderer | null>(null);

const canvasSize = {
    width: 500,
    height: 500
};

// 监听数据变化
watch([() => props.trainingPoints, () => props.testPoints, () => props.config], () => {
    if (renderer.value && canvasRef.value) {
        renderer.value.render(
            props.trainingPoints,
            props.testPoints,
            {
                showDecisionBoundary: props.config.showDecisionBoundary,
                showConfidence: props.config.showConfidence,
                animateDistances: props.config.animateDistances,
                enhancedMode: props.config.enhancedMode,
                classColors,
                useWeightedKNN: props.config.useWeightedKNN,
                k: props.config.k,
                distanceMetric: props.config.distanceMetric,
            }
        );
    }
}, { deep: true });

// 初始化画布
onMounted(() => {
    if (canvasRef.value) {
        renderer.value = new CanvasRenderer(canvasRef.value);
        renderer.value.startAnimation();
        
        // 初始渲染
        if (props.trainingPoints.length > 0) {
            renderer.value.render(
                props.trainingPoints,
                props.testPoints,
                {
                    showDecisionBoundary: props.config.showDecisionBoundary,
                    showConfidence: props.config.showConfidence,
                    animateDistances: props.config.animateDistances,
                    enhancedMode: props.config.enhancedMode,
                    classColors,
                    useWeightedKNN: props.config.useWeightedKNN,
                    k: props.config.k,
                    distanceMetric: props.config.distanceMetric,
                }
            );
        }
    }
});


// 清理动画
onUnmounted(() => {
    if (renderer.value) {
        renderer.value.stopAnimation();
    }
});

const handlePointerDown = (event: PointerEvent) => {
    if (!canvasRef.value || !renderer.value) return;
    // 阻止默认行为（如手机上长按弹出菜单）
    event.preventDefault();
    // 获取精确的 Canvas 坐标（关键！）
    const rect = canvasRef.value.getBoundingClientRect();
    const scaleX = canvasRef.value.width / rect.width;
    const scaleY = canvasRef.value.height / rect.height;

    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    const normalizedPoint = {
        x: x / canvasRef.value.width,
        y: y / canvasRef.value.height
    };

    emit('point-added', normalizedPoint);
};

</script>

<style scoped>
.canvas-container {
    width: 100%;
}

canvas {
    width: 100%;
    max-width: 500px;
    margin: 0 auto;
    border: 1px solid #d1d5db;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    background-color: white;
    cursor: crosshair;
    aspect-ratio: 1 / 1;
    transition: box-shadow 0.3s;
}

canvas:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

/* 添加缺失的CSS变量 */
:root {
    --blue-500: #3b82f6;
    --red-500: #ef4444;
    --green-500: #10b981;
    --yellow-500: #f59e0b;
    --purple-500: #8b5cf6;
    --gray-900: #111827;
    --gray-800: #1f2937;
    --gray-700: #374151;
    --gray-300: #d1d5db;
    --gray-200: #e5e7eb;
    --gray-100: #f3f4f6;
    --blue-600: #2563eb;
    --blue-100: #dbeafe;
    --indigo-50: #eef2ff;
}
</style>