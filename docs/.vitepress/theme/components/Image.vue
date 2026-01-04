<template>
    <div :class="[
        'my-image-wrapper',
        alignClass,
        { 'my-image-card': card }
    ]" :style="{ width: width }">
        <img v-bind="$attrs" :src="src" :alt="alt" class="my-image-content" />
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps({
    src: {
        type: String,
        required: true
    },
    alt: {
        type: String,
        default: ''
    },
    width: {
        type: String,
        default: '100%' // 支持 '50%', '300px', 'auto' 等
    },
    align: {
        type: String,
        default: 'center',
        validator: (value: string) => ['left', 'center', 'right'].includes(value)
    },
    card: {
        type: Boolean,
        default: false
    }
})

const alignClass = computed(() => {
    return `my-image-align-${props.align}`
})
</script>

<style scoped>
/* 基础容器 */
.my-image-wrapper {
    margin-bottom: 1rem;
}

/* 对齐控制 */
.my-image-align-left {
    float: left;
    margin-right: 1rem;
}

.my-image-align-right {
    float: right;
    margin-left: 1rem;
}

.my-image-align-center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    text-align: center;
}

/* Material 风格卡片 */
.my-image-card {
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow:
        0 2px 4px rgba(0, 0, 0, 0.1),
        0 8px 16px rgba(0, 0, 0, 0.1);
    transition: box-shadow 0.2s ease;
}

.my-image-card:hover {
    box-shadow:
        0 4px 8px rgba(0, 0, 0, 0.15),
        0 12px 24px rgba(0, 0, 0, 0.2);
}

/* 图片样式 */
.my-image-content {
    width: 100%;
    height: auto;
    display: block;
    max-width: 100%;
}
</style>