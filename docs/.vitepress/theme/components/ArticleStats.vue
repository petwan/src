<template>
    <div class="article-stats" v-if="wordCount > 0">
        <span>字数: {{ wordCount }}</span>
        <span>· 阅读时长: {{ readingTime }} 分钟</span>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const wordCount = ref(0)
const readingTime = ref(0)

// 中文阅读速度约 300～500 字/分钟，英文约 200～250 词/分钟
// 这里我们按中文为主，取 400 字/分钟
const WORDS_PER_MINUTE = 400

onMounted(() => {
    // 获取主内容区域（VitePress 默认内容容器）
    const contentEl = document.querySelector('.VPContent')

    if (contentEl) {
        // 提取所有文本，去除多余空白
        const text = contentEl.innerText || ''
        // 统计中英文字符（包括汉字、字母、数字）
        const matches = text.match(/[\u4e00-\u9fa5a-zA-Z0-9]+/g)
        const count = matches ? matches.join('').length : 0

        wordCount.value = count
        readingTime.value = Math.ceil(count / WORDS_PER_MINUTE)
    }
})
</script>

<style scoped>
.article-stats {
    margin-top: 1rem;
    padding-bottom: 1rem;
    font-size: 0.85em;
    color: var(--vp-c-text-2);
    display: flex;
    gap: 0.5em;
}
</style>