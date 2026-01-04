<template>
  <nav class="breadcrumb" aria-label="breadcrumb">
    <ol>
      <li><a href="/">首页</a></li>
      <li v-for="(item, index) in crumbs" :key="index">
        <a v-if="!item.active" :href="item.path">{{ item.name }}</a>
        <span v-else>{{ item.name }}</span>
      </li>
    </ol>
  </nav>
</template>

<script setup>
import { useRoute } from 'vitepress'
import { computed } from 'vue'

const route = useRoute()
const crumbs = computed(() => {
  const path = route.path.replace(/\/+$/, '').replace(/\.html$/, '')
  if (path === '/') return []
  const segments = path.split('/').filter(Boolean)
  return segments.map((seg, i) => {
    const isLast = i === segments.length - 1
    const p = '/' + segments.slice(0, i + 1).join('/')
    const name = decodeURIComponent(seg)
      .replace(/-/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
    return { name, path: p, active: isLast }
  })
})
</script>

<style scoped>
/* 关键：使用 flex 保证同行 */
.breadcrumb ol {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.25rem;
  /* 替代 ::after 的 margin，更现代 */
  padding: 0;
  margin: 0 0 1rem;
  list-style: none;
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
}

.breadcrumb a {
  color: var(--vp-c-brand);
  text-decoration: none;
}

.breadcrumb a:hover {
  text-decoration: underline;
}

/* 使用 gap 后，可以用这个替代分隔符 */
.breadcrumb ol li:not(:first-child)::before {
  content: '/';
  color: var(--vp-c-text-3);
  margin-right: 0.25rem;
}
</style>