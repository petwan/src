<style scoped>
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
}

.dot {
    position: absolute;
    left: 0.25rem;
    top: 0.25rem;
    background-color: white;
    width: 1rem;
    height: 1rem;
    border-radius: 9999px;
    transition: transform 0.3s;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.translate-x-4 {
    transform: translateX(1rem);
}

.panel {
    background-color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.block {
    display: block;
}

.w-full {
    width: 100%;
}

.p-2 {
    padding: 0.5rem;
}

.border {
    border-width: 1px;
}

.border-gray-300 {
    border-color: #d1d5db;
}

.rounded-md {
    border-radius: 0.375rem;
}

.mb-4 {
    margin-bottom: 1rem;
}

.text-sm {
    font-size: 0.875rem;
    line-height: 1.25rem;
}

.font-medium {
    font-weight: 500;
}

.flex {
    display: flex;
}

.items-center {
    align-items: center;
}

.cursor-pointer {
    cursor: pointer;
}

.relative {
    position: relative;
}

.w-10 {
    width: 2.5rem;
}

.h-6 {
    height: 1.5rem;
}

.bg-gray-300 {
    background-color: #d1d5db;
}

.bg-blue-600 {
    background-color: #2563eb;
}

.ml-3 {
    margin-left: 0.75rem;
}

.mt-1 {
    margin-top: 0.25rem;
}

.text-xs {
    font-size: 0.75rem;
    line-height: 1rem;
}

.text-gray-500 {
    color: #6b7280;
}

.justify-between {
    justify-content: space-between;
}

.flex-wrap {
    flex-wrap: wrap;
}

.h-2 {
    height: 0.5rem;
}

.appearance-none {
    appearance: none;
}

.bg-gray-200 {
    background-color: #e5e7eb;
}

.rounded-lg {
    border-radius: 0.5rem;
}

.mb-1 {
    margin-bottom: 0.25rem;
}

.mb-3 {
    margin-bottom: 0.75rem;
}

.text-lg {
    font-size: 1.125rem;
    line-height: 1.75rem;
}

.font-semibold {
    font-weight: 600;
}

.mt-4 {
    margin-top: 1rem;
}

.text-gray-600 {
    color: #4b5563;
}

.py-2 {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}

.text-white {
    color: white;
}

.transition {
    transition: all 0.3s;
}

.bg-gray-100 {
    background-color: #f3f4f6;
}

.rounded {
    border-radius: 0.25rem;
}

/* Enhanced select and button styling matching HTML */
.control-select {
    padding: 0.5rem;
    border: 1px solid #d1d5db;
    border-radius: 0.375rem;
    width: 100%;
    background-color: white;
    color: #374151;
}

.control-button {
    width: 100%;
    padding: 0.5rem 1rem;
    background-color: #2563eb;
    color: white;
    border-radius: 0.375rem;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s;
    font-weight: 500;
    font-size: 0.875rem;
    line-height: 1.25rem;
}

.control-button:hover {
    background-color: #1d4ed8;
}

/* Enhanced hover states */
.hover\:bg-blue-700:hover {
    background-color: #1d4ed8;
}

/* Fix for switch visibility */
input[type="checkbox"].sr-only:checked+div.block {
    background-color: #2563eb;
}

input[type="checkbox"].sr-only+div.block {
    background-color: #d1d5db;
}

input[type="checkbox"].sr-only:checked+div.block+.dot {
    transform: translateX(1rem);
}

input[type="checkbox"].sr-only+div.block {
    position: relative;
    display: inline-block;
    width: 2.5rem;
    height: 1.5rem;
    border-radius: 9999px;
}

input[type="checkbox"].sr-only+div.block+.dot {
    position: absolute;
    left: 0.25rem;
    top: 0.25rem;
    background-color: white;
    width: 1rem;
    height: 1rem;
    border-radius: 9999px;
    transition: transform 0.3s;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}
</style>

<template>
    <div class="control-panel panel">

        <!-- K的取值配置 -->
        <div class="mb-4">
            <label class="block text-sm font-medium mb-1">
                K的取值: <span id="kValueText">{{ config.k }}</span>
            </label>
            <input type="range" :value="config.k"
                @input="updateConfig('k', Number(($event.target as HTMLInputElement).value))" min="1" max="15"
                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer" />
            <div class="flex justify-between text-xs text-gray-500 mt-1">
                <span>1</span>
                <span>15</span>
            </div>
        </div>

        <!-- 距离计算 -->
        <div class="mb-4">
            <label class="block text-sm font-medium mb-1">距离计算</label>
            <select :value="config.distanceMetric"
                @change="updateConfig('distanceMetric', ($event.target as HTMLSelectElement).value as 'euclidean' | 'manhattan')"
                class="control-select w-full p-2 border border-gray-300 rounded-md">
                <option value="euclidean">Euclidean Distance</option>
                <option value="manhattan">Manhattan Distance</option>
            </select>
        </div>

        <!-- 加权KNN选择 -->
        <div class="mb-4 flex items-center">
            <label class="flex items-center cursor-pointer">
                <div class="relative">
                    <input type="checkbox" :checked="config.useWeightedKNN"
                        @change="updateConfig('useWeightedKNN', !config.useWeightedKNN)" class="sr-only">
                    <div class="block w-10 h-6 rounded-full"
                        :class="config.useWeightedKNN ? 'bg-blue-600' : 'bg-gray-300'"></div>
                    <div class="dot" :class="config.useWeightedKNN ? 'translate-x-4' : ''"></div>
                </div>
                <div class="ml-3 text-sm font-medium">加权Knn</div>
            </label>
        </div>

        <!-- 显示概率 -->
        <div class="mb-4 flex items-center">
            <label class="flex items-center cursor-pointer">
                <div class="relative">
                    <input type="checkbox" :checked="config.showConfidence"
                        @change="updateConfig('showConfidence', !config.showConfidence)" class="sr-only">
                    <div class="block w-10 h-6 rounded-full"
                        :class="config.showConfidence ? 'bg-blue-600' : 'bg-gray-300'"></div>
                    <div class="dot" :class="config.showConfidence ? 'translate-x-4' : ''"></div>
                </div>
                <div class="ml-3 text-sm font-medium">显示Confience</div>
            </label>
        </div>

        <!-- 决策边界 -->
        <div class="mb-4 flex items-center">
            <label class="flex items-center cursor-pointer">
                <div class="relative">
                    <input type="checkbox" :checked="config.showDecisionBoundary"
                        @change="updateConfig('showDecisionBoundary', !config.showDecisionBoundary)" class="sr-only">
                    <div class="block w-10 h-6 rounded-full"
                        :class="config.showDecisionBoundary ? 'bg-blue-600' : 'bg-gray-300'"></div>
                    <div class="dot" :class="config.showDecisionBoundary ? 'translate-x-4' : ''"></div>
                </div>
                <div class="ml-3 text-sm font-medium">显示决策边界</div>
            </label>
        </div>

        <!-- 数据集选择 -->
        <div class="mb-4">
            <label class="block text-sm font-medium mb-1">数据集选择</label>
            <select :value="config.selectedDataset"
                @change="updateConfig('selectedDataset', ($event.target as HTMLSelectElement).value)"
                class="control-select w-full p-2 border border-gray-300 rounded-md">
                <option v-for="(dataset, key) in datasets" :key="key" :value="key">
                    {{ dataset.name }}
                </option>
            </select>
        </div>


        <!-- 清除测试点 -->
        <button @click="clearTestPoints"
            class="control-button py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition w-full">
            清除测试点
        </button>
    </div>
</template>

<script lang="ts">
import { defineComponent, PropType } from 'vue';

export default defineComponent({
    props: {
        config: {
            type: Object as PropType<{
                k: number;
                distanceMetric: 'euclidean' | 'manhattan';
                useWeightedKNN: boolean;
                showConfidence: boolean;
                showDecisionBoundary: boolean;
                selectedDataset: string;
            }>,
            required: true,
        },
        datasets: {
            type: Object as PropType<{
                [key: string]: {
                    name: string;
                    description: string;
                };
            }>,
            required: true,
        },
    },
    methods: {
        updateConfig(key: string, value: any) {
            this.$emit('update:config', {
                ...this.config,
                [key]: value,
            });
        },
        clearTestPoints() {
            this.$emit('clear-test-points');
        },
    },
});
</script>