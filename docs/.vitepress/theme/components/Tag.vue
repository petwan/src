<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'

const { frontmatter } = useData()

const tags = computed(() => {
  const fmTags = frontmatter.value.tags
  if (!fmTags) return []
  return Array.isArray(fmTags) ? fmTags : [fmTags]
})
</script>

<template>
  <div v-if="tags.length > 0" class="tags-container">
    <span 
      v-for="(tag, index) in tags" 
      :key="index" 
      class="tag"
    >
      {{ tag }}
    </span>
  </div>
</template>

<style scoped>
.tags-container {
  margin: 12px 0;
}

.tag {
  display: inline-block;
  padding: 4px 10px;
  margin-right: 8px;
  background-color: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 600;
}

.tag:hover {
  background-color: var(--vp-c-brand-1);
  color: var(--vp-c-bg-elv);
}
</style>