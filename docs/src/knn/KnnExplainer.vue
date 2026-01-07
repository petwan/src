<template>
  <div class="knn-explainer">

    <div class="main-content">
      <!-- 左侧：画布 -->
      <div class="left-panel">
        <KnnCanvas :training-points="currentDataset.points" :test-points="testPoints" :config="localConfig"
          @point-added="addTestPoint" />
      </div>

      <!-- 中间：控制面板 -->
      <div class="center-panel">
        <ControlPanel :config="localConfig" :datasets="datasets" @update:config="updateConfig"
          @clear-test-points="clearTestPoints" />
      </div>

    </div>
  </div>
</template>


<style scoped>
/* 基础容器 */
.knn-explainer {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1rem;
}

/* 头部样式 */
.header {
  text-align: center;
  margin-bottom: 2rem;
}

.title {
  font-size: 2.25rem;
  font-weight: 700;
  background: linear-gradient(to right, #2563eb, #7c3aed);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  margin-bottom: 1rem;
}

.subtitle {
  font-size: 1.125rem;
  color: #4b5563;
  margin-bottom: 1.5rem;
}

/* 主要内容区域 */
.main-content {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* 在桌面端改为水平布局 */
@media (min-width: 768px) {
  .main-content {
    flex-direction: row;
    justify-content: center;
    gap: 8px;
    /* 8px间距 */
  }

  .left-panel {
    flex: 0 0 75%;
    /* 25%宽度 */
    max-width: 700px;
  }

  .center-panel {
    flex: 0 0 25%;
    /* 50%宽度 */
    max-width: 600px;
  }

  .right-panel {
    flex: 0 0 25%;
    /* 25%宽度 */
    max-width: 400px;
  }
}

/* 移动端样式 */
@media (max-width: 767px) {
  .main-content {
    gap: 16px;
  }

  .left-panel,
  .center-panel,
  .right-panel {
    width: 100%;
    max-width: 100%;
  }
}

/* 面板内部样式 */
.left-panel,
.center-panel,
.right-panel {
  display: flex;
  justify-content: center;
}

/* 可选：添加背景和边框增强视觉效果 */
.left-panel,
.center-panel,
.right-panel {
  padding: 1rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.explanation-box {
  margin-top: 1rem;
  padding: 1rem;
  border-radius: 0.5rem;
}

.explanation-box h3 {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
}

.explanation-box p {
  margin-bottom: 1rem;
  line-height: 1.6;
}
</style>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { generateDatasets, classColors } from './core/datasets'
import { KNNEngine } from './core/knn-engine'
import type { Point, TestPoint, KNNConfig, Dataset } from './core/types'
import KnnCanvas from './components/KnnCanvas.vue'
import ControlPanel from './components/ControlPanel.vue'

// 初始化数据集
const datasets = generateDatasets()

// 初始化配置
const localConfig = ref<KNNConfig>({
  k: 3,
  distanceMetric: 'euclidean',
  useWeightedKNN: false,
  showConfidence: true,
  showDecisionBoundary: false,
  animateDistances: false,
  enhancedMode: true,
  selectedDataset: 'twoClassBasic',
  gameMode: true
})

// 当前数据集
const currentDataset = computed((): Dataset => datasets[localConfig.value.selectedDataset])

// 测试点列表
const testPoints = ref<TestPoint[]>([])

// KNN引擎实例
const knnEngine = new KNNEngine()

// 计算平均置信度
const avgConfidence = computed(() => {
  if (testPoints.value.length === 0) return 0
  const totalConfidence = testPoints.value.reduce((sum, point) => {
    return sum + (point.confidence || 0)
  }, 0)
  return Math.round((totalConfidence / testPoints.value.length) * 100)
})

// 总分计算（简化版）
const totalScore = computed(() => {
  return Math.min(100, Math.floor(avgConfidence.value * 0.5 + testPoints.value.length * 2))
})

// 更新配置
const updateConfig = (newConfig: KNNConfig) => {
  localConfig.value = newConfig;
}

// 添加测试点
const addTestPoint = (normalizedPoint: { x: number; y: number }) => {
  const newPoint: TestPoint = {
    id: Date.now().toString(),
    x: normalizedPoint.x,
    y: normalizedPoint.y,
    class: -1 // 占位符，稍后会被分类结果替换
  }

  // 分类新点
  const result = knnEngine.classifyPoint(
    newPoint,
    localConfig.value.k,
    localConfig.value.distanceMetric,
    localConfig.value.useWeightedKNN
  )

  // 更新点的信息
  const classifiedPoint: TestPoint = {
    ...newPoint,
    predictedClass: result.predictedClass,
    confidence: result.confidence,
    nearestNeighbors: result.neighbors
  }

  testPoints.value.push(classifiedPoint)
}

// 清除测试点
const clearTestPoints = () => {
  testPoints.value = []
}

// 组件挂载时初始化训练点
onMounted(() => {
  knnEngine.setTrainingPoints(currentDataset.value.points)
})

// 监听数据集变化
watch(() => localConfig.value.selectedDataset, () => {
  // 数据集改变时清除测试点
  clearTestPoints()
  // 更新KNN引擎的训练点
  knnEngine.setTrainingPoints(currentDataset.value.points)
})

// 监听配置变化
watch(localConfig, () => {
  // 当K值、距离度量或加权选项改变时，重新分类所有测试点
  testPoints.value = testPoints.value.map(point => {
    const result = knnEngine.classifyPoint(
      point,
      localConfig.value.k,
      localConfig.value.distanceMetric,
      localConfig.value.useWeightedKNN
    )

    return {
      ...point,
      predictedClass: result.predictedClass,
      confidence: result.confidence,
      nearestNeighbors: result.neighbors
    }
  })
}, { deep: true })
</script>
